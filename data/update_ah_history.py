"""Pre-download A/H research data from official/public market endpoints.

All network access in this module is opt-in through :func:`update_ah_history`
or the ``update`` CLI command.  Importing the module and all normalization /
loading helpers are offline.

The published A/H pair table is a *current live cohort*, not a point-in-time
historical universe.  The published metadata deliberately labels the resulting
survivorship bias.  No historical ``valid_from`` value is invented.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ah_history"
PAIR_FILENAME = "ah_pairs.parquet"
HK_PRICE_FILENAME = "hk_prices.parquet"
FX_FILENAME = "hk_connect_settlement_fx.parquet"
METADATA_FILENAME = "metadata.json"
CHECKPOINT_DIRNAME = "hk_checkpoints"

EASTMONEY_AH_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_AH_FALLBACK_URLS = (
    "https://82.push2.eastmoney.com/api/qt/clist/get",
    "https://48.push2.eastmoney.com/api/qt/clist/get",
    "https://72.push2.eastmoney.com/api/qt/clist/get",
    "http://push2.eastmoney.com/api/qt/clist/get",
    "http://82.push2.eastmoney.com/api/qt/clist/get",
)
TENCENT_HK_URL = "https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
TENCENT_HK_RAW_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
TENCENT_AH_LIST_URL = "http://stock.gtimg.cn/data/hk_rank.php"
HKEX_EQUITY_PAGE_URL = (
    "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities/Equities-Quote"
)
HKEX_EQUITY_WIDGET_URL = "https://www1.hkex.com.hk/hkexwidget/data/getequityquote"
GITHUB_AH_REGISTRY_URL = (
    "https://raw.githubusercontent.com/xcnecon/"
    "A-H-Premium-Arbitrage-Monitor/refs/heads/main/ah_pairs.csv"
)
SSE_FX_URL = "https://query.sse.com.cn/commonSoaQuery.do"
SZSE_FX_URL = "https://www.szse.cn/api/report/ShowReport"

PAIR_COLUMNS = (
    "a_code",
    "h_code",
    "name",
    "snapshot_date",
    "universe_cohort",
    "point_in_time_complete",
    "survivorship_bias",
    "share_unit_assumption",
)
HK_SOURCE_COLUMNS = (
    "date",
    "h_code",
    "raw_open",
    "raw_close",
    "raw_high",
    "raw_low",
    "qfq_open",
    "qfq_close",
    "qfq_high",
    "qfq_low",
    "hfq_open",
    "hfq_close",
    "volume",
)
HK_DERIVED_COLUMNS = (
    "qfq_scale",
    "qfq_intercept",
    "qfq_affine_valid",
    "h_pre_close",
)
HK_PRICE_COLUMNS = (*HK_SOURCE_COLUMNS, *HK_DERIVED_COLUMNS)
FX_COLUMNS = (
    "date",
    "exchange",
    "buy_rate",
    "sell_rate",
    "mid_rate",
    "unit",
)
COHORT_LABEL = "current_live_cohort"
SHARE_UNIT_ASSUMPTION = "ordinary_share_1_to_1_unverified"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".parquet", delete=False
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        round_trip = pd.read_parquet(temporary)
        if len(round_trip) != len(frame) or list(round_trip.columns) != list(
            frame.columns
        ):
            raise RuntimeError(f"parquet round-trip changed shape/schema for {path.name}")
        digest = _sha256_file(temporary)
        os.replace(temporary, path)
        temporary = None
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def canonical_a_code(value: object) -> str:
    """Return a six-digit SH/SZ code without accepting an ambiguous exchange."""

    text = str(value).strip().upper()
    suffix = re.fullmatch(r"(\d{1,6})\.(SH|SZ)", text)
    supplied_exchange = None
    if suffix:
        text, supplied_exchange = suffix.groups()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    if not re.fullmatch(r"\d{1,6}", text):
        raise ValueError(f"invalid A-share code: {value!r}")
    digits = text.zfill(6)
    if digits[0] in "569":
        exchange = "SH"
    elif digits[0] in "0123":
        exchange = "SZ"
    else:
        raise ValueError(f"unsupported A-share exchange: {value!r}")
    if supplied_exchange is not None and supplied_exchange != exchange:
        raise ValueError(f"mismatched A-share exchange: {value!r}")
    return f"{digits}.{exchange}"


def canonical_h_code(value: object) -> str:
    """Return a five-digit Hong Kong code with the explicit ``.HK`` suffix."""

    text = str(value).strip().upper()
    suffix = re.fullmatch(r"(\d{1,5})\.HK", text)
    if suffix:
        text = suffix.group(1)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    if not re.fullmatch(r"\d{1,5}", text):
        raise ValueError(f"invalid H-share code: {value!r}")
    return f"{text.zfill(5)}.HK"


def eastmoney_page_count(payload: Mapping[str, object], page_size: int) -> int:
    """Compute the number of Eastmoney pages from its reported total."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("Eastmoney response is missing data")
    total = data.get("total")
    try:
        total_int = int(total)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Eastmoney response is missing a valid total") from exc
    if total_int < 0 or page_size <= 0:
        raise RuntimeError("invalid Eastmoney pagination values")
    return max(1, math.ceil(total_int / page_size))


def _eastmoney_rows(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("Eastmoney response is missing data")
    diff = data.get("diff")
    if isinstance(diff, Mapping):
        rows = list(diff.values())
    elif isinstance(diff, list):
        rows = diff
    elif diff is None:
        rows = []
    else:
        raise RuntimeError("Eastmoney response has an invalid diff payload")
    if not all(isinstance(row, Mapping) for row in rows):
        raise RuntimeError("Eastmoney diff contains a non-object row")
    return rows


def normalize_ah_pairs(
    rows: Iterable[Mapping[str, object]], *, snapshot_date: object
) -> pd.DataFrame:
    """Normalize a current Eastmoney A/H board snapshot.

    Eastmoney's comparison/premium fields (including ``f188``/``f189``) are
    intentionally ignored: they are price-comparison fields, not economic share
    conversion ratios.
    """

    normalized: list[dict[str, object]] = []
    snapshot = pd.Timestamp(snapshot_date).normalize()
    if pd.isna(snapshot):
        raise ValueError("snapshot_date is invalid")
    for row in rows:
        try:
            a_code = canonical_a_code(row.get("f191"))
            h_code = canonical_h_code(row.get("f12"))
        except ValueError as exc:
            raise RuntimeError(f"invalid Eastmoney A/H row: {row!r}") from exc
        name = str(row.get("f193") or row.get("f14") or "").strip()
        if not name:
            raise RuntimeError(f"Eastmoney A/H row has no name: {row!r}")
        normalized.append(
            {
                "a_code": a_code,
                "h_code": h_code,
                "name": name,
                "snapshot_date": snapshot,
                "universe_cohort": COHORT_LABEL,
                "point_in_time_complete": False,
                "survivorship_bias": True,
                "share_unit_assumption": SHARE_UNIT_ASSUMPTION,
            }
        )
    frame = pd.DataFrame(normalized, columns=PAIR_COLUMNS)
    if frame.empty:
        raise RuntimeError("Eastmoney A/H snapshot is empty")
    frame = frame.drop_duplicates(["a_code", "h_code"], keep="last")
    frame = frame.sort_values(["a_code", "h_code"], kind="stable").reset_index(
        drop=True
    )
    _validate_pairs(frame)
    return frame


def _validate_pairs(frame: pd.DataFrame) -> None:
    missing = set(PAIR_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"A/H pairs are missing columns: {sorted(missing)}")
    if "valid_from" in frame.columns:
        raise RuntimeError("current A/H snapshots must not fabricate valid_from")
    if frame.empty:
        raise RuntimeError("A/H pair table is empty")
    if not frame["a_code"].astype(str).str.fullmatch(r"\d{6}\.(SH|SZ)").all():
        raise RuntimeError("A/H pair table contains an invalid A-share code")
    if not frame["h_code"].astype(str).str.fullmatch(r"\d{5}\.HK").all():
        raise RuntimeError("A/H pair table contains an invalid H-share code")
    if frame.duplicated(["a_code", "h_code"]).any():
        raise RuntimeError("A/H pair table contains duplicate pairs")
    if not (frame["universe_cohort"] == COHORT_LABEL).all():
        raise RuntimeError("A/H universe cohort label is missing")
    if frame["point_in_time_complete"].astype(bool).any():
        raise RuntimeError("current-live A/H data cannot be marked PIT complete")
    if not frame["survivorship_bias"].astype(bool).all():
        raise RuntimeError("current-live A/H data must disclose survivorship bias")


@dataclass
class _RateLimiter:
    delay_seconds: float
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    _last_call: float | None = None

    def wait(self) -> None:
        now = self.monotonic()
        if self._last_call is not None:
            remaining = self.delay_seconds - (now - self._last_call)
            if remaining > 0:
                self.sleep(remaining)
        self._last_call = self.monotonic()


def _get_response(
    session: object,
    url: str,
    *,
    params: Mapping[str, object],
    headers: Mapping[str, str],
    limiter: _RateLimiter,
    timeout: int = 60,
    attempts: int = 3,
) -> object:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        limiter.wait()
        try:
            response = session.get(
                url, params=dict(params), headers=dict(headers), timeout=timeout
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                limiter.sleep(float(2**attempt))
    assert last_error is not None
    raise last_error


def download_current_ah_pairs(
    session: object,
    *,
    snapshot_date: object,
    delay_seconds: float = 1.1,
) -> tuple[pd.DataFrame, int]:
    """Serially download every page of the current Eastmoney A/H board.

    Eastmoney serves the same API from numbered ``push2`` frontends.  A failed
    frontend is retried from the first page on an alternate host so a partial
    page set can never be mixed across attempts.
    """

    page_size = 100
    errors: list[Exception] = []
    for endpoint in (EASTMONEY_AH_URL, *EASTMONEY_AH_FALLBACK_URLS):
        page = 1
        page_count = 1
        rows: list[Mapping[str, object]] = []
        limiter = _RateLimiter(delay_seconds)
        try:
            while page <= page_count:
                params = {
                    "np": "1",
                    "fltt": "1",
                    "invt": "2",
                    "fs": "b:DLMK0101",
                    "fields": "f193,f191,f12,f14",
                    "fid": "f3",
                    "pn": str(page),
                    "pz": str(page_size),
                    "po": "1",
                    "dect": "1",
                }
                response = _get_response(
                    session,
                    endpoint,
                    params=params,
                    headers={
                        "User-Agent": "Mozilla/5.0 (WBR A/H updater)",
                        "Referer": "https://quote.eastmoney.com/",
                    },
                    limiter=limiter,
                )
                payload = response.json()
                if page == 1:
                    page_count = eastmoney_page_count(payload, page_size)
                rows.extend(_eastmoney_rows(payload))
                page += 1
            frame = normalize_ah_pairs(rows, snapshot_date=snapshot_date)
            frame.attrs["mapping_source"] = "eastmoney_current_ah_board"
            frame.attrs["candidate_h_codes"] = int(frame["h_code"].nunique())
            frame.attrs["unresolved_h_codes"] = []
            return frame, page_count
        except Exception as exc:
            errors.append(exc)
    warnings.warn(
        "All Eastmoney A/H frontends failed; falling back first to a dated "
        "open-source current registry whose rows disclose HKEX/manual provenance, "
        "then to Tencent candidates plus HKEX underlying_ric if needed.",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        frame = download_github_current_ah_pairs(
            session,
            snapshot_date=snapshot_date,
            delay_seconds=delay_seconds,
        )
        frame.attrs["eastmoney_errors"] = [
            f"{type(exc).__name__}: {exc}" for exc in errors
        ]
        return frame, 0
    except Exception as github_error:
        errors.append(github_error)
    try:
        frame, tencent_pages = download_hkex_current_ah_pairs(
            session,
            snapshot_date=snapshot_date,
            delay_seconds=delay_seconds,
        )
        frame.attrs["eastmoney_errors"] = [
            f"{type(exc).__name__}: {exc}" for exc in errors
        ]
        return frame, -tencent_pages
    except Exception as fallback_error:
        raise RuntimeError(
            "all Eastmoney frontends, the open-source registry, and the "
            "Tencent/HKEX official-mapping "
            "fallback failed: "
            + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            + f"; fallback={type(fallback_error).__name__}: {fallback_error}"
        ) from fallback_error


def normalize_github_ah_pairs(
    payload: bytes,
    *,
    snapshot_date: object,
) -> pd.DataFrame:
    """Normalize the Apache-2.0 current registry without treating it as PIT."""

    source = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    required = {
        "hk_code",
        "a_code",
        "name",
        "status",
        "source",
        "first_seen",
    }
    missing = required.difference(source.columns)
    if missing:
        raise RuntimeError(f"GitHub A/H registry schema changed: {sorted(missing)}")
    active = source.loc[source["status"].str.lower() == "active"].copy()
    rows = [
        {"f191": row.a_code, "f12": row.hk_code, "f193": row.name}
        for row in active.itertuples(index=False)
    ]
    frame = normalize_ah_pairs(rows, snapshot_date=snapshot_date)
    if len(frame) < 100:
        raise RuntimeError(
            f"GitHub current A/H registry is implausibly small: {len(frame)}"
        )
    frame.attrs["mapping_source"] = "open_source_current_registry"
    frame.attrs["registry_url"] = GITHUB_AH_REGISTRY_URL
    frame.attrs["registry_sha256"] = _sha256_bytes(payload)
    frame.attrs["registry_total_rows"] = int(len(source))
    frame.attrs["candidate_h_codes"] = int(len(active))
    frame.attrs["unresolved_h_codes"] = []
    frame.attrs["registry_source_breakdown"] = {
        str(key): int(value)
        for key, value in active["source"].str.lower().value_counts().items()
    }
    frame.attrs["registry_first_seen_min"] = str(active["first_seen"].min())
    frame.attrs["registry_first_seen_max"] = str(active["first_seen"].max())
    return frame


def download_github_current_ah_pairs(
    session: object,
    *,
    snapshot_date: object,
    delay_seconds: float = 1.1,
) -> pd.DataFrame:
    """Download one dated current registry snapshot as a recoverable fallback."""

    response = _get_response(
        session,
        GITHUB_AH_REGISTRY_URL,
        params={},
        headers={"User-Agent": "Mozilla/5.0 (WBR A/H updater)"},
        limiter=_RateLimiter(delay_seconds),
    )
    return normalize_github_ah_pairs(response.content, snapshot_date=snapshot_date)


def _lenient_json_object(text: str) -> Mapping[str, object]:
    """Decode a JSON/JSONP/JavaScript-assignment object from a public endpoint."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("response contains no object")
    body = text[start : end + 1]
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        # Tencent's legacy rank endpoint occasionally emits JavaScript object
        # syntax.  AKShare already carries the compatible decoder.
        from akshare.utils import demjson

        value = demjson.decode(body)
    if not isinstance(value, Mapping):
        raise RuntimeError("response root is not an object")
    return value


def parse_tencent_ah_candidates(text: str) -> tuple[list[dict[str, str]], int]:
    """Parse H-code candidates from Tencent's current A_H ranking pages."""

    payload = _lenient_json_object(text)
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("Tencent A/H response is missing data")
    try:
        page_count = int(data.get("page_count"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Tencent A/H response has no page_count") from exc
    page_data = data.get("page_data") or []
    if not isinstance(page_data, list):
        raise RuntimeError("Tencent A/H page_data is not a list")
    rows: list[dict[str, str]] = []
    for item in page_data:
        packed = item[0] if isinstance(item, (list, tuple)) and item else item
        if not isinstance(packed, str):
            raise RuntimeError("Tencent A/H page contains an invalid row")
        fields = packed.split("~")
        if len(fields) < 2:
            raise RuntimeError("Tencent A/H packed row has fewer than two fields")
        rows.append(
            {
                "h_code": canonical_h_code(fields[0].removeprefix("hk")),
                "name": fields[1].strip(),
            }
        )
    return rows, page_count


def extract_hkex_token(html: str) -> str:
    """Extract the rotating token from the HKEX equity quote page."""

    block = re.search(
        r"(?s)LabCI\.getToken\s*=\s*function\s*\(\)\s*\{.*?"
        r"return\s+[\"'](evLtsLsBNAUVTPxtGqVe[^\"']+)[\"']\s*;",
        html,
    )
    if block:
        return block.group(1)
    simple = re.search(r"evLtsLsBNAUVTPxtGqVe[^\"'<>\s]+", html)
    if simple:
        return simple.group(0)
    raise RuntimeError("HKEX quote page contains no current LabCI token")


def parse_hkex_underlying_quote(text: str) -> dict[str, str] | None:
    """Parse one official HKEX widget quote into an A/H mapping."""

    payload = _lenient_json_object(text)
    data = payload.get("data")
    if not isinstance(data, Mapping) or str(data.get("responsecode")) != "000":
        return None
    quote = data.get("quote")
    if not isinstance(quote, Mapping):
        return None
    underlying = str(quote.get("underlying_ric") or "").strip().upper()
    match = re.fullmatch(r"(\d{6})\.(SS|SZ)", underlying)
    if not match:
        return None
    digits, provider_exchange = match.groups()
    a_code = f"{digits}.{'SH' if provider_exchange == 'SS' else 'SZ'}"
    return {
        "a_code": canonical_a_code(a_code),
        "h_code": canonical_h_code(quote.get("ric")),
        "name": str(quote.get("nm") or "").strip(),
    }


def download_hkex_current_ah_pairs(
    session: object,
    *,
    snapshot_date: object,
    delay_seconds: float = 1.1,
) -> tuple[pd.DataFrame, int]:
    """Resolve Tencent's current A/H H codes with HKEX official mappings."""

    headers = {
        "User-Agent": "Mozilla/5.0 (WBR A/H updater)",
        "Referer": "https://www.hkex.com.hk/",
    }
    limiter = _RateLimiter(delay_seconds)
    first = _get_response(
        session,
        TENCENT_AH_LIST_URL,
        params={
            "board": "A_H",
            "metric": "price",
            "pageSize": "20",
            "reqPage": "0",
            "order": "decs",
            "var_name": "list_data",
        },
        headers=headers,
        limiter=limiter,
    )
    first_rows, page_count = parse_tencent_ah_candidates(first.text)
    candidates = list(first_rows)
    for page in range(1, page_count):
        response = _get_response(
            session,
            TENCENT_AH_LIST_URL,
            params={
                "board": "A_H",
                "metric": "price",
                "pageSize": "20",
                "reqPage": str(page),
                "order": "decs",
                "var_name": "list_data",
            },
            headers=headers,
            limiter=limiter,
        )
        page_rows, reported_pages = parse_tencent_ah_candidates(response.text)
        if reported_pages != page_count:
            raise RuntimeError("Tencent A/H page_count changed during download")
        candidates.extend(page_rows)
    candidate_by_h = {
        row["h_code"]: row for row in candidates if row.get("h_code")
    }
    if len(candidate_by_h) < 50:
        raise RuntimeError(
            f"Tencent current A/H candidate list is implausibly small: {len(candidate_by_h)}"
        )

    mappings: list[dict[str, str]] = []
    unresolved: list[str] = []
    for h_code, candidate in sorted(candidate_by_h.items()):
        symbol = str(int(h_code[:5]))
        page_response = _get_response(
            session,
            HKEX_EQUITY_PAGE_URL,
            params={"sc_lang": "en", "sym": symbol},
            headers=headers,
            limiter=limiter,
        )
        token = extract_hkex_token(page_response.text)
        milliseconds = int(time.time() * 1000)
        callback = f"jQuery351{milliseconds}"
        widget_response = _get_response(
            session,
            HKEX_EQUITY_WIDGET_URL,
            params={
                "sym": symbol,
                # The page embeds an already percent-encoded token.  Decode it
                # once so requests performs exactly one URL-encoding pass.
                "token": unquote(token),
                "lang": "eng",
                "qid": str(milliseconds),
                "callback": callback,
                "_": str(milliseconds),
            },
            headers={**headers, "Referer": page_response.url},
            limiter=limiter,
        )
        mapping = parse_hkex_underlying_quote(widget_response.text)
        if mapping is None or mapping["h_code"] != h_code:
            unresolved.append(h_code)
            continue
        if not mapping["name"]:
            mapping["name"] = candidate["name"]
        mappings.append(
            {"f191": mapping["a_code"], "f12": h_code, "f193": mapping["name"]}
        )
    if len(mappings) < 50:
        raise RuntimeError(
            f"HKEX resolved only {len(mappings)} of {len(candidate_by_h)} A/H candidates"
        )
    frame = normalize_ah_pairs(mappings, snapshot_date=snapshot_date)
    frame.attrs["mapping_source"] = "tencent_current_candidates_plus_hkex_underlying_ric"
    frame.attrs["candidate_h_codes"] = len(candidate_by_h)
    frame.attrs["unresolved_h_codes"] = unresolved
    return frame, page_count


def _json_from_jsonp(text: str) -> Mapping[str, object]:
    """Decode Tencent JSON or JavaScript-assignment/JSONP payloads."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Tencent response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, Mapping):
        raise RuntimeError("Tencent response root is not an object")
    return value


def parse_tencent_hk_kline(
    payload: str | Mapping[str, object],
    *,
    h_code: object,
    adjusted: bool,
    adjustment: str = "qfq",
) -> pd.DataFrame:
    """Parse Tencent ``day``/``qfqday``/``hfqday`` rows into a price frame."""

    data_json = _json_from_jsonp(payload) if isinstance(payload, str) else payload
    canonical = canonical_h_code(h_code)
    provider_code = f"hk{canonical[:5]}"
    data = data_json.get("data")
    if data in (None, []):
        if data_json.get("code") in (0, "0"):
            return pd.DataFrame(columns=("date", "open", "close", "volume"))
        raise RuntimeError(
            f"Tencent returned code={data_json.get('code')!r}: "
            f"{data_json.get('msg')!r}"
        )
    if not isinstance(data, Mapping):
        raise RuntimeError("Tencent response is missing data")
    stock = data.get(provider_code)
    if stock is None:
        return pd.DataFrame(columns=("date", "open", "close", "volume"))
    if not isinstance(stock, Mapping):
        raise RuntimeError("Tencent stock payload is not an object")
    if adjustment not in {"qfq", "hfq"}:
        raise ValueError("adjustment must be qfq or hfq")
    row_key = f"{adjustment}day" if adjusted else "day"
    rows = stock.get(row_key)
    if rows is None:
        return pd.DataFrame(columns=("date", "open", "close", "volume"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Tencent {row_key} payload is not a list")
    parsed: list[tuple[object, object, object, object, object, object]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise RuntimeError(f"Tencent {row_key} contains a malformed row")
        parsed.append((row[0], row[1], row[2], row[3], row[4], row[5]))
    frame = pd.DataFrame(
        parsed,
        columns=("date", "open", "close", "high", "low", "volume"),
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "close", "high", "low", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "open", "close", "high", "low"]].isna().any().any():
        raise RuntimeError(f"Tencent {row_key} contains invalid dates/prices")
    # Tencent qfq histories may contain negative synthetic dividend-adjusted
    # levels.  They are retained because the research layer uses *differences*
    # normalized by the same-day affine raw-to-qfq scale, never qfq percentage
    # returns.  Raw and hfq OHLC must remain positive.
    require_positive = (not adjusted) or adjustment == "hfq"
    if require_positive and (
        frame[["open", "close", "high", "low"]] <= 0
    ).any().any():
        raise RuntimeError(f"Tencent {row_key} contains non-positive prices")
    frame = frame.drop_duplicates("date", keep="last")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def align_hk_price_frames(
    raw: pd.DataFrame,
    qfq: pd.DataFrame,
    hfq: pd.DataFrame,
    *,
    h_code: object,
) -> pd.DataFrame:
    """Outer-align raw, qfq, and hfq histories without filling prices."""

    raw_columns = raw.rename(
        columns={
            "open": "raw_open",
            "close": "raw_close",
            "high": "raw_high",
            "low": "raw_low",
            "volume": "raw_volume",
        }
    )
    qfq_columns = qfq.rename(
        columns={
            "open": "qfq_open",
            "close": "qfq_close",
            "high": "qfq_high",
            "low": "qfq_low",
            "volume": "qfq_volume",
        }
    )
    hfq_columns = hfq.rename(
        columns={
            "open": "hfq_open",
            "close": "hfq_close",
            "volume": "hfq_volume",
        }
    )
    frame = raw_columns.merge(
        qfq_columns, on="date", how="outer", validate="one_to_one", sort=True
    )
    frame = frame.merge(
        hfq_columns.loc[:, ["date", "hfq_open", "hfq_close", "hfq_volume"]],
        on="date",
        how="outer",
        validate="one_to_one",
        sort=True,
    )
    frame["h_code"] = canonical_h_code(h_code)
    raw_volume = pd.to_numeric(frame.pop("raw_volume"), errors="coerce")
    qfq_volume = pd.to_numeric(frame.pop("qfq_volume"), errors="coerce")
    hfq_volume = pd.to_numeric(frame.pop("hfq_volume"), errors="coerce")
    frame["volume"] = raw_volume.where(
        raw_volume.notna(), qfq_volume.where(qfq_volume.notna(), hfq_volume)
    )
    for column in HK_SOURCE_COLUMNS[2:]:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[:, HK_SOURCE_COLUMNS]
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    return frame


def _date_chunks(start_date: object, end_date: object) -> list[tuple[date, date]]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start_date must not be later than end_date")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        # Two calendar years contain fewer than Tencent's 640-row limit even
        # around leap years, while keeping the full update request count modest.
        chunk_end = min(date(cursor.year + 1, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = date(chunk_end.year + 1, 1, 1)
    return chunks


def _download_one_hk_history(
    session: object,
    *,
    h_code: str,
    start_date: object,
    end_date: object,
    limiter: _RateLimiter,
) -> pd.DataFrame:
    raw_parts: list[pd.DataFrame] = []
    qfq_parts: list[pd.DataFrame] = []
    hfq_parts: list[pd.DataFrame] = []
    provider_code = f"hk{canonical_h_code(h_code)[:5]}"
    headers = {
        "User-Agent": "Mozilla/5.0 (WBR A/H updater)",
        "Referer": f"https://gu.qq.com/{provider_code}/gp",
    }
    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        for adjustment, parts in (
            ("raw", raw_parts),
            ("qfq", qfq_parts),
            ("hfq", hfq_parts),
        ):
            adjusted = adjustment != "raw"
            adjust = "" if adjustment == "raw" else adjustment
            params = {
                "param": (
                    f"{provider_code},day,{chunk_start.isoformat()},"
                    f"{chunk_end.isoformat()},640,{adjust}"
                )
            }
            response = _get_response(
                session,
                TENCENT_HK_URL if adjusted else TENCENT_HK_RAW_URL,
                params=params,
                headers=headers,
                limiter=limiter,
            )
            parts.append(
                parse_tencent_hk_kline(
                    response.text,
                    h_code=h_code,
                    adjusted=adjusted,
                    adjustment=adjustment if adjusted else "qfq",
                )
            )
    raw_parts = [part for part in raw_parts if not part.empty]
    qfq_parts = [part for part in qfq_parts if not part.empty]
    hfq_parts = [part for part in hfq_parts if not part.empty]
    raw = (
        pd.concat(raw_parts, ignore_index=True)
        if raw_parts
        else pd.DataFrame(columns=("date", "open", "close", "high", "low", "volume"))
    )
    qfq = (
        pd.concat(qfq_parts, ignore_index=True)
        if qfq_parts
        else pd.DataFrame(columns=("date", "open", "close", "high", "low", "volume"))
    )
    hfq = (
        pd.concat(hfq_parts, ignore_index=True)
        if hfq_parts
        else pd.DataFrame(columns=("date", "open", "close", "high", "low", "volume"))
    )
    raw = raw.drop_duplicates("date", keep="last").sort_values("date")
    qfq = qfq.drop_duplicates("date", keep="last").sort_values("date")
    hfq = hfq.drop_duplicates("date", keep="last").sort_values("date")
    frame = attach_h_preclose(
        align_hk_price_frames(raw, qfq, hfq, h_code=h_code)
    )
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    frame = frame[frame["date"].between(requested_start, requested_end)].reset_index(
        drop=True
    )
    if frame.empty:
        raise RuntimeError(f"Tencent returned no history for {h_code}")
    return frame


def _validate_hk_prices(frame: pd.DataFrame, *, allow_empty: bool = False) -> None:
    missing = set(HK_PRICE_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"HK price data is missing columns: {sorted(missing)}")
    if frame.empty and not allow_empty:
        raise RuntimeError("HK price data is empty")
    if not frame.empty:
        if not frame["h_code"].astype(str).str.fullmatch(r"\d{5}\.HK").all():
            raise RuntimeError("HK price data contains invalid codes")
        if frame.duplicated(["h_code", "date"]).any():
            raise RuntimeError("HK price data contains duplicate code/date rows")
        raw_columns = (
            "raw_open",
            "raw_close",
            "raw_high",
            "raw_low",
        )
        raw = frame.loc[:, raw_columns].apply(pd.to_numeric, errors="coerce")
        raw_values = raw.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(raw_values).all() or (raw_values <= 0.0).any():
            raise RuntimeError("HK raw prices must be finite and positive")
        # Tencent does not publish hfq history for every security.  hfq is kept
        # solely as a discontinuity audit series, so an entirely missing row is
        # allowed; a partially missing or non-positive published row is not.
        hfq = frame.loc[:, ("hfq_open", "hfq_close")].apply(
            pd.to_numeric, errors="coerce"
        )
        hfq_missing = hfq.isna()
        if (hfq_missing.any(axis=1) != hfq_missing.all(axis=1)).any():
            raise RuntimeError("HK hfq rows must be either complete or absent")
        published_hfq = ~hfq_missing.all(axis=1)
        hfq_values = hfq.loc[published_hfq].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(hfq_values).all() or (hfq_values <= 0.0).any():
            raise RuntimeError("published HK hfq prices must be finite and positive")
        published_valid = frame["qfq_affine_valid"].fillna(False).astype(bool)
        scale = pd.to_numeric(frame["qfq_scale"], errors="coerce")
        intercept = pd.to_numeric(frame["qfq_intercept"], errors="coerce")
        if (
            scale.loc[published_valid].isna().any()
            or (scale.loc[published_valid] <= 0.0).any()
            or intercept.loc[published_valid].isna().any()
        ):
            raise RuntimeError("valid qfq affine rows need finite parameters")
        pre_close = pd.to_numeric(frame["h_pre_close"], errors="coerce")
        if (pre_close.dropna() <= 0.0).any():
            raise RuntimeError("causal H preClose must be positive when present")
        if pre_close.notna().any() and not published_valid.loc[
            pre_close.notna()
        ].all():
            raise RuntimeError("causal H preClose cannot use an invalid affine row")


def qfq_affine_parameters(
    frame: pd.DataFrame,
    *,
    residual_tolerance: float = 0.0021,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit each day's affine ``qfq = scale * raw + intercept`` mapping.

    Tencent applies one affine corporate-action transform to all four OHLC
    fields on a date.  The positive slope converts a raw share price into the
    qfq economic-unit scale.  Qfq levels themselves may be negative after many
    dividends, so callers must use differences divided by this slope rather
    than percentage changes in qfq levels.

    Returns ``(scale, intercept, valid)``.  A flat raw OHLC row has no
    identifiable slope and is invalid; callers must break the adjustment chain
    instead of filling it from a future row.
    """

    raw_columns = ("raw_open", "raw_close", "raw_high", "raw_low")
    qfq_columns = ("qfq_open", "qfq_close", "qfq_high", "qfq_low")
    missing = set((*raw_columns, *qfq_columns)).difference(frame.columns)
    if missing:
        raise ValueError(f"H history missing affine columns: {sorted(missing)}")
    raw = frame.loc[:, raw_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    qfq = frame.loc[:, qfq_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    raw_mean = np.mean(raw, axis=1)
    qfq_mean = np.mean(qfq, axis=1)
    raw_centered = raw - raw_mean[:, None]
    qfq_centered = qfq - qfq_mean[:, None]
    denominator = np.sum(raw_centered * raw_centered, axis=1)
    numerator = np.sum(raw_centered * qfq_centered, axis=1)
    scale = np.full(len(frame), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(numerator, denominator, out=scale, where=denominator > 0.0)
    intercept = qfq_mean - scale * raw_mean
    predicted = scale[:, None] * raw + intercept[:, None]
    residual = np.max(np.abs(qfq - predicted), axis=1)
    ordered_raw = np.sort(raw, axis=1)
    distinct = 1 + np.sum(np.diff(ordered_raw, axis=1) > 1e-9, axis=1)
    raw_span = np.ptp(raw, axis=1)
    valid = (
        np.all(np.isfinite(raw), axis=1)
        & np.all(np.isfinite(qfq), axis=1)
        & np.isfinite(scale)
        & (scale > 0.0)
        & np.isfinite(intercept)
        & np.isfinite(residual)
        & (residual <= residual_tolerance)
        & (distinct >= 3)
        & (raw_span >= 0.02 - 1e-12)
    )
    scale[~valid] = np.nan
    intercept[~valid] = np.nan
    return scale, intercept, valid


def attach_h_preclose(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a causal corporate-action-adjusted previous close.

    For consecutive *identifiable* Tencent qfq mappings
    ``Q_t(x) = a_t*x + b_t`` this reconstructs the raw-unit ex-action
    reference price

    ``preClose_t = (a_prev*rawClose_prev + b_prev - b_t) / a_t``.

    Future affine rebasing ``Q' = uQ + v`` cancels algebraically.  Invalid or
    underdetermined daily mappings are never backfilled from the future; the
    next identifiable row links to the last identifiable completed close.
    """

    values = frame.copy()
    if values.empty:
        for column in HK_DERIVED_COLUMNS:
            values[column] = pd.Series(dtype="bool" if column.endswith("valid") else "float64")
        return values.loc[:, HK_PRICE_COLUMNS]
    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()
    values = values.sort_values(["h_code", "date"], kind="stable").reset_index(
        drop=True
    )
    scale, intercept, affine_valid = qfq_affine_parameters(values)
    raw_close = pd.to_numeric(values["raw_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    pre_close = np.full(len(values), np.nan, dtype=np.float64)
    codes = values["h_code"].astype(str).to_numpy()
    previous_index: int | None = None
    for row in range(len(values)):
        if row == 0 or codes[row] != codes[row - 1]:
            previous_index = None
        if not affine_valid[row] or not np.isfinite(raw_close[row]):
            continue
        if previous_index is not None:
            previous = previous_index
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                candidate = (
                    scale[previous] * raw_close[previous]
                    + intercept[previous]
                    - intercept[row]
                ) / scale[row]
            if np.isfinite(candidate) and candidate > 0.0:
                pre_close[row] = candidate
        previous_index = row
    values["qfq_scale"] = scale
    values["qfq_intercept"] = intercept
    values["qfq_affine_valid"] = affine_valid
    values["h_pre_close"] = pre_close
    return values.loc[:, HK_PRICE_COLUMNS]


def hfq_discontinuity_mask(frame: pd.DataFrame) -> np.ndarray:
    """Flag implausible hfq jumps not present in the corresponding raw close."""

    required = {"h_code", "date", "raw_close", "hfq_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"H history missing hfq audit columns: {sorted(missing)}")
    values = frame.loc[:, ["h_code", "date", "raw_close", "hfq_close"]].copy()
    values["_position"] = np.arange(len(values), dtype=np.intp)
    values["raw_close"] = pd.to_numeric(values["raw_close"], errors="coerce")
    values["hfq_close"] = pd.to_numeric(values["hfq_close"], errors="coerce")
    ordered = values.sort_values(["h_code", "date"], kind="stable")
    same_code = ordered["h_code"].eq(ordered["h_code"].shift())
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_move = np.log(ordered["raw_close"] / ordered["raw_close"].shift())
        hfq_move = np.log(ordered["hfq_close"] / ordered["hfq_close"].shift())
    flagged = (
        same_code
        & raw_move.abs().lt(math.log(1.25))
        & hfq_move.abs().gt(math.log(1.50))
    )
    result = np.zeros(len(frame), dtype=bool)
    result[ordered["_position"].to_numpy(dtype=np.intp)] = flagged.to_numpy(
        dtype=bool
    )
    return result


def download_hk_histories(
    session: object,
    pairs: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    start_date: object,
    end_date: object,
    resume: bool = True,
    delay_seconds: float = 0.15,
) -> tuple[pd.DataFrame, int]:
    """Download Tencent histories serially, checkpointing one H share at a time."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_label = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_label = pd.Timestamp(end_date).strftime("%Y%m%d")
    limiter = _RateLimiter(delay_seconds)
    parts: list[pd.DataFrame] = []
    resumed = 0
    for h_code in sorted(pairs["h_code"].astype(str).unique()):
        digits = canonical_h_code(h_code)[:5]
        checkpoint = checkpoint_dir / (
            f"{digits}_{start_label}_{end_label}_raw_qfq_hfq_ohlc_v3.parquet"
        )
        if resume and checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
            _validate_hk_prices(frame)
            if set(frame["h_code"].astype(str)) != {canonical_h_code(h_code)}:
                raise RuntimeError(f"checkpoint code mismatch: {checkpoint}")
            resumed += 1
        else:
            frame = _download_one_hk_history(
                session,
                h_code=h_code,
                start_date=start_date,
                end_date=end_date,
                limiter=limiter,
            )
            _atomic_write_parquet(frame, checkpoint)
        parts.append(frame)
    parts = [part for part in parts if not part.empty]
    if not parts:
        raise RuntimeError("all Tencent H-share histories are empty")
    # Concatenate the already validated fixed schema explicitly.  Some older
    # Tencent series have an entirely missing volume column; generic
    # DataFrame.concat otherwise emits an all-NA dtype inference warning whose
    # behavior is scheduled to change in pandas.
    combined = pd.DataFrame(
        {
            column: np.concatenate(
                [part[column].to_numpy(copy=False) for part in parts]
            )
            for column in HK_PRICE_COLUMNS
        },
        columns=HK_PRICE_COLUMNS,
    )
    combined = combined.drop_duplicates(["h_code", "date"], keep="last")
    combined = combined.sort_values(["h_code", "date"], kind="stable").reset_index(
        drop=True
    )
    for column in (
        *HK_SOURCE_COLUMNS[2:],
        "qfq_scale",
        "qfq_intercept",
        "h_pre_close",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["qfq_affine_valid"] = (
        combined["qfq_affine_valid"].fillna(False).astype(bool)
    )
    _validate_hk_prices(combined)
    return combined, resumed


def sse_page_count(payload: Mapping[str, object]) -> int:
    """Read SSE's authoritative ``pageHelp.pageCount`` field."""

    page_help = payload.get("pageHelp")
    if not isinstance(page_help, Mapping):
        raise RuntimeError("SSE response is missing pageHelp")
    try:
        count = int(page_help.get("pageCount"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SSE response has no valid pageCount") from exc
    if count < 1:
        raise RuntimeError("SSE pageCount must be positive")
    return count


def normalize_fx_rows(
    rows: Iterable[Mapping[str, object]], *, exchange: str
) -> pd.DataFrame:
    """Normalize and deduplicate settlement FX as CNY paid per one HKD."""

    exchange = exchange.upper()
    if exchange not in {"SH", "SZ"}:
        raise ValueError("exchange must be SH or SZ")
    source = pd.DataFrame(rows)
    required = {"validDate", "buyPrice", "sellPrice"}
    missing = required.difference(source.columns)
    if missing:
        raise RuntimeError(f"FX source is missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(source["validDate"], errors="coerce").dt.normalize(),
            "exchange": exchange,
            "buy_rate": pd.to_numeric(source["buyPrice"], errors="coerce"),
            "sell_rate": pd.to_numeric(source["sellPrice"], errors="coerce"),
        }
    )
    if frame[["date", "buy_rate", "sell_rate"]].isna().any().any():
        raise RuntimeError("FX source contains invalid dates or rates")
    frame = frame.drop_duplicates(["exchange", "date"], keep="last")
    frame["mid_rate"] = (frame["buy_rate"] + frame["sell_rate"]) / 2.0
    frame["unit"] = "CNY/HKD"
    frame = frame.loc[:, FX_COLUMNS]
    frame = frame.sort_values(["exchange", "date"], kind="stable").reset_index(
        drop=True
    )
    _validate_fx(frame)
    return frame


def normalize_szse_fx_workbook(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official SZSE ``SGT_LSHL/tab2`` XLSX worksheet."""

    source = source.dropna(axis="columns", how="all")
    candidates = {
        "validDate": ("适用日期",),
        "buyPrice": ("买入结算汇兑比率", "结算汇率买入价"),
        "sellPrice": ("卖出结算汇兑比率", "结算汇率卖出价"),
    }
    selected = {
        target: next((name for name in names if name in source.columns), None)
        for target, names in candidates.items()
    }
    missing = [target for target, name in selected.items() if name is None]
    if missing:
        raise RuntimeError(
            f"SZSE settlement workbook schema changed; missing={sorted(missing)}, "
            f"actual={list(source.columns)}"
        )
    if "货币种类" in source.columns:
        source = source[source["货币种类"].astype(str).str.upper() == "HKD"]
    aliases = {name: target for target, name in selected.items() if name is not None}
    renamed = source.rename(columns=aliases)
    return normalize_fx_rows(renamed.to_dict("records"), exchange="SZ")


def _validate_fx(frame: pd.DataFrame) -> None:
    missing = set(FX_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"FX data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("FX data is empty")
    if not set(frame["exchange"].astype(str)).issubset({"SH", "SZ"}):
        raise RuntimeError("FX data contains an invalid exchange")
    if frame.duplicated(["exchange", "date"]).any():
        raise RuntimeError("FX data contains duplicate exchange/date rows")
    if not (frame["unit"] == "CNY/HKD").all():
        raise RuntimeError("FX data has an incorrect unit")
    rates = frame[["buy_rate", "sell_rate", "mid_rate"]]
    if not ((rates > 0.5) & (rates < 1.5)).all().all():
        raise RuntimeError("FX rate is outside the plausible CNY/HKD unit range")
    expected_mid = (frame["buy_rate"] + frame["sell_rate"]) / 2.0
    if not (frame["mid_rate"].sub(expected_mid).abs() < 1e-12).all():
        raise RuntimeError("FX mid rate is inconsistent with buy/sell rates")


def download_sse_settlement_fx(
    session: object,
    *,
    end_date: object,
    delay_seconds: float = 0.5,
) -> tuple[pd.DataFrame, int]:
    """Download all SSE settlement-rate pages using ``pageHelp.pageCount``."""

    page = 1
    pages = 1
    rows: list[Mapping[str, object]] = []
    limiter = _RateLimiter(delay_seconds)
    while page <= pages:
        params = {
            "isPagination": "true",
            "updateDate": "20120601",
            "updateDateEnd": pd.Timestamp(end_date).strftime("%Y%m%d"),
            "sqlId": "FW_HGT_JSHDBL",
            "pageHelp.cacheSize": "1",
            "pageHelp.pageSize": "2000",
            "pageHelp.pageNo": str(page),
            "pageHelp.beginPage": str(page),
            "pageHelp.endPage": str(page),
        }
        response = _get_response(
            session,
            SSE_FX_URL,
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0 (WBR A/H updater)",
                "Referer": "https://www.sse.com.cn/",
            },
            limiter=limiter,
        )
        payload = response.json()
        if page == 1:
            pages = sse_page_count(payload)
        result = payload.get("result")
        if not isinstance(result, list):
            raise RuntimeError("SSE response result is not a list")
        rows.extend(result)
        page += 1
    return normalize_fx_rows(rows, exchange="SH"), pages


def download_szse_settlement_fx(
    session: object, *, delay_seconds: float = 0.5
) -> tuple[pd.DataFrame, str]:
    """Download the official SZSE full-history settlement-rate XLSX."""

    limiter = _RateLimiter(delay_seconds)
    response = _get_response(
        session,
        SZSE_FX_URL,
        params={
            "SHOWTYPE": "xlsx",
            "CATALOGID": "SGT_LSHL",
            "TABKEY": "tab2",
        },
        headers={
            "User-Agent": "Mozilla/5.0 (WBR A/H updater)",
            "Referer": "https://www.szse.cn/szhk/hkbussiness/exchangerate/",
        },
        limiter=limiter,
    )
    payload = response.content
    if not payload.startswith(b"PK\x03\x04"):
        raise RuntimeError("SZSE settlement response is not an XLSX workbook")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style.*",
            category=UserWarning,
        )
        source = pd.read_excel(io.BytesIO(payload), engine="openpyxl")
    return normalize_szse_fx_workbook(source), _sha256_bytes(payload)


def _coverage(frame: pd.DataFrame, date_column: str = "date") -> dict[str, object]:
    dates = pd.to_datetime(frame[date_column])
    return {
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
    }


def _schema(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(frame[column].dtype) for column in frame.columns}


def build_metadata(
    *,
    pairs: pd.DataFrame,
    hk_prices: pd.DataFrame,
    fx: pd.DataFrame,
    hashes: Mapping[str, str],
    snapshot_date: object,
    requested_start_date: object,
    requested_end_date: object,
    eastmoney_pages: int,
    sse_pages: int,
    resumed_h_codes: int,
    szse_source_sha256: str | None,
    szse_limitation: str | None,
) -> dict[str, object]:
    """Build the auditable manifest; this function is pure and offline."""

    exchanges = sorted(fx["exchange"].astype(str).unique().tolist())
    metadata: dict[str, object] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": pd.Timestamp(snapshot_date).date().isoformat(),
        "urls": {
            "current_ah_comparison": EASTMONEY_AH_URL,
            "current_ah_candidates_fallback": TENCENT_AH_LIST_URL,
            "official_hkex_mapping_fallback": HKEX_EQUITY_WIDGET_URL,
            "open_source_current_registry_fallback": GITHUB_AH_REGISTRY_URL,
            "hk_raw_daily": TENCENT_HK_RAW_URL,
            "hk_qfq_daily": TENCENT_HK_URL,
            "hk_hfq_daily": TENCENT_HK_URL,
            "sse_settlement_fx": SSE_FX_URL,
            "szse_settlement_fx": SZSE_FX_URL,
        },
        "source_parameters": {
            "eastmoney_board": "b:DLMK0101",
            "sse_sql_id": "FW_HGT_JSHDBL",
            "szse_catalog_id": "SGT_LSHL",
            "szse_tab_key": "tab2",
        },
        "universe": {
            "cohort": COHORT_LABEL,
            "point_in_time_complete": False,
            "survivorship_bias": True,
            "warning": (
                "The published pair table is a current live A/H snapshot. "
                "Historical tests using this cohort are survivor-biased and are "
                "not an unbiased point-in-time universe test."
            ),
        },
        "pair_mapping": {
            "source": pairs.attrs.get(
                "mapping_source", "eastmoney_current_ah_board"
            ),
            "candidate_h_codes": int(
                pairs.attrs.get("candidate_h_codes", pairs["h_code"].nunique())
            ),
            "resolved_pairs": int(len(pairs)),
            "unresolved_h_codes": list(
                pairs.attrs.get("unresolved_h_codes", [])
            ),
            "eastmoney_errors": list(pairs.attrs.get("eastmoney_errors", [])),
            "registry_sha256": pairs.attrs.get("registry_sha256"),
            "registry_total_rows": pairs.attrs.get("registry_total_rows"),
            "registry_source_breakdown": dict(
                pairs.attrs.get("registry_source_breakdown", {})
            ),
            "registry_first_seen": {
                "min": pairs.attrs.get("registry_first_seen_min"),
                "max": pairs.attrs.get("registry_first_seen_max"),
            },
        },
        "share_unit_assumption": {
            "label": SHARE_UNIT_ASSUMPTION,
            "a_ordinary_shares_per_unit": 1.0,
            "h_ordinary_shares_per_unit": 1.0,
            "issuer_specific_terms_verified": False,
            "eastmoney_price_ratio_fields_used": False,
            "warning": (
                "The 1:1 ordinary-share economic unit is an explicit research "
                "assumption; verify issuer-specific terms before executable use."
            ),
        },
        "fx_contract": {
            "unit": "CNY/HKD",
            "meaning": "CNY paid per one HKD",
            "routing": {"SH": "SSE settlement rate", "SZ": "SZSE settlement rate"},
            "available_exchanges": exchanges,
            "limitation": szse_limitation,
        },
        "request": {
            "hk_start_date": pd.Timestamp(requested_start_date).date().isoformat(),
            "hk_end_date": pd.Timestamp(requested_end_date).date().isoformat(),
        },
        "pagination": {
            "eastmoney_pages": max(0, eastmoney_pages),
            "tencent_fallback_pages": max(0, -eastmoney_pages),
            "sse_pages": sse_pages,
            "serial_requests": True,
        },
        "resume": {
            "checkpoint_directory": CHECKPOINT_DIRNAME,
            "resumed_h_codes": resumed_h_codes,
        },
        "artifacts": {
            PAIR_FILENAME: {
                "sha256": hashes[PAIR_FILENAME],
                "rows": len(pairs),
                "unique_a_codes": int(pairs["a_code"].nunique()),
                "unique_h_codes": int(pairs["h_code"].nunique()),
                "schema": _schema(pairs),
                "coverage": {
                    "snapshot_date": pd.to_datetime(pairs["snapshot_date"])
                    .max()
                    .date()
                    .isoformat()
                },
            },
            HK_PRICE_FILENAME: {
                "sha256": hashes[HK_PRICE_FILENAME],
                "rows": len(hk_prices),
                "unique_h_codes": int(hk_prices["h_code"].nunique()),
                "schema": _schema(hk_prices),
                "coverage": _coverage(hk_prices),
                "nonpositive_qfq_rows": int(
                    (
                        (hk_prices["qfq_open"] <= 0)
                        | (hk_prices["qfq_close"] <= 0)
                    ).sum()
                ),
                "qfq_affine_invalid_rows": int(
                    (~hk_prices["qfq_affine_valid"].astype(bool)).sum()
                ),
                "codes_without_affine_rows": sorted(
                    hk_prices.groupby("h_code", sort=True)["qfq_affine_valid"]
                    .any()
                    .loc[lambda values: ~values]
                    .index.astype(str)
                    .tolist()
                ),
                "causal_preclose_rows": int(hk_prices["h_pre_close"].notna().sum()),
                "hfq_discontinuity_rows": int(hfq_discontinuity_mask(hk_prices).sum()),
                "adjusted_contract": (
                    "Research uses raw OHLC plus a causal h_pre_close reconstructed "
                    "from the same-day raw/qfq four-price affine map. It never logs "
                    "or percentage-changes qfq/hfq levels. Non-positive qfq levels "
                    "are valid affine inputs; underdetermined/non-affine rows fail "
                    "closed. Hfq is retained only for discontinuity audit."
                ),
            },
            FX_FILENAME: {
                "sha256": hashes[FX_FILENAME],
                "rows": len(fx),
                "schema": _schema(fx),
                "coverage_by_exchange": {
                    exchange: _coverage(fx[fx["exchange"] == exchange])
                    for exchange in exchanges
                },
            },
        },
        "source_hashes": {"szse_xlsx_sha256": szse_source_sha256},
    }
    return metadata


def update_ah_history(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: object = "2014-11-17",
    end_date: object | None = None,
    resume: bool = True,
    allow_sse_only: bool = False,
    eastmoney_delay: float = 1.1,
    tencent_delay: float = 0.05,
    exchange_delay: float = 0.5,
    session: object | None = None,
) -> dict[str, object]:
    """Explicit network update entry point; publish metadata after all artifacts."""

    if end_date is None:
        end_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not be later than end_date")
    if min(eastmoney_delay, tencent_delay, exchange_delay) < 0:
        raise ValueError("request delays must be non-negative")

    if session is None:
        import requests

        session = requests.Session()

    snapshot_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, eastmoney_pages = download_current_ah_pairs(
        session,
        snapshot_date=snapshot_date,
        delay_seconds=eastmoney_delay,
    )
    hk_prices, resumed_h_codes = download_hk_histories(
        session,
        pairs,
        checkpoint_dir=output_dir / CHECKPOINT_DIRNAME,
        start_date=start_date,
        end_date=end_date,
        resume=resume,
        delay_seconds=tencent_delay,
    )
    sse_fx, sse_pages = download_sse_settlement_fx(
        session, end_date=end_date, delay_seconds=exchange_delay
    )
    szse_limitation = None
    try:
        szse_fx, szse_source_sha256 = download_szse_settlement_fx(
            session, delay_seconds=exchange_delay
        )
    except Exception as exc:
        if not allow_sse_only:
            raise RuntimeError(
                "SZSE settlement FX download failed; no SSE rate was silently "
                "substituted for SZ pairs. Re-run with --allow-sse-only only for "
                "an explicitly limited SH-only artifact."
            ) from exc
        szse_fx = pd.DataFrame(columns=FX_COLUMNS)
        szse_source_sha256 = None
        szse_limitation = (
            "SZSE settlement FX was unavailable. This artifact is SH-only; SZ "
            "pairs must be excluded, never valued with the SSE rate. Error: "
            f"{type(exc).__name__}: {exc}"
        )
    fx = (
        pd.concat([sse_fx, szse_fx], ignore_index=True)
        if not szse_fx.empty
        else sse_fx.copy()
    )
    fx = fx.sort_values(["exchange", "date"], kind="stable").reset_index(drop=True)
    _validate_fx(fx)

    hashes = {
        PAIR_FILENAME: _atomic_write_parquet(pairs, output_dir / PAIR_FILENAME),
        HK_PRICE_FILENAME: _atomic_write_parquet(
            hk_prices, output_dir / HK_PRICE_FILENAME
        ),
        FX_FILENAME: _atomic_write_parquet(fx, output_dir / FX_FILENAME),
    }
    metadata = build_metadata(
        pairs=pairs,
        hk_prices=hk_prices,
        fx=fx,
        hashes=hashes,
        snapshot_date=snapshot_date,
        requested_start_date=start_date,
        requested_end_date=end_date,
        eastmoney_pages=eastmoney_pages,
        sse_pages=sse_pages,
        resumed_h_codes=resumed_h_codes,
        szse_source_sha256=szse_source_sha256,
        szse_limitation=szse_limitation,
    )
    _atomic_write_json(metadata, output_dir / METADATA_FILENAME)
    return metadata


def load_ah_history(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load and hash-verify the three research artifacts entirely offline."""

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for filename in (PAIR_FILENAME, HK_PRICE_FILENAME, FX_FILENAME):
        path = output_dir / filename
        expected = metadata["artifacts"][filename]["sha256"]
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"A/H artifact SHA-256 mismatch for {filename}: "
                f"expected={expected} actual={actual}"
            )
        frames.append(pd.read_parquet(path))
    pairs, hk_prices, fx = frames
    _validate_pairs(pairs)
    _validate_hk_prices(hk_prices)
    _validate_fx(fx)
    return pairs, hk_prices, fx, metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="download and publish A/H data")
    update.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    update.add_argument(
        "--start",
        default="2014-11-17",
        help="HK history start; default matches the first SSE settlement-FX date",
    )
    update.add_argument("--end", default=None)
    update.add_argument("--no-resume", action="store_true")
    update.add_argument(
        "--allow-sse-only",
        action="store_true",
        help=(
            "publish an explicitly SH-only FX artifact if SZSE is unavailable; "
            "SZ pairs must then be excluded"
        ),
    )
    update.add_argument("--eastmoney-delay", type=float, default=1.1)
    update.add_argument("--tencent-delay", type=float, default=0.05)
    update.add_argument("--exchange-delay", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "update":
        metadata = update_ah_history(
            output_dir=args.output_dir,
            start_date=args.start,
            end_date=args.end,
            resume=not args.no_resume,
            allow_sse_only=args.allow_sse_only,
            eastmoney_delay=args.eastmoney_delay,
            tencent_delay=args.tencent_delay,
            exchange_delay=args.exchange_delay,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

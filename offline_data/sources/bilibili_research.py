"""Download public Bilibili video metadata and subtitles for offline research.

This module deliberately uses only Bilibili's public JSON endpoints.  It does
not require cookies, login state, or private credentials.  The command line
entry point writes raw responses as UTF-8 JSON so the research snapshot can be
audited and replayed without network access.
"""

from __future__ import annotations

from utils.atomic_file import file_sha256
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OWNER_UID = 133578883
BVIDS = [
    "BV16MY46tEDG", "BV1Twtn66EbX", "BV1638968EzE", "BV1dd876XEoc",
    "BV1U2gP6XERd", "BV17HMk6qEqa", "BV1bx3C6eE12", "BV1SXg66sENU",
    "BV1G6Mv6GERx", "BV1CNLQ6REu5", "BV1h2Vm6WEQv", "BV1iwG16QEvJ",
    "BV1FkL16KEpQ", "BV1wS516qEhB", "BV12zRWBLEuX", "BV1Pc9SB5EJn",
    "BV1dRopBAEzT", "BV1sWQ7BHExm", "BV1iyDpBEEGn", "BV1tQ95BpEEa",
]
API_BASE = "https://api.bilibili.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WBR-research/1.0)",
    "Accept": "application/json",
}


def _request_json(path: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    url = f"{API_BASE}{path}?{urlencode(params)}"
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    return url, json.loads(body)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _error_record(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _download(url: str, path: Path, bvid: str) -> str:
    request_headers = {**HEADERS, "Referer": f"https://www.bilibili.com/video/{bvid}/", "Origin": "https://www.bilibili.com"}
    request = Request(url, headers=request_headers)
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=60) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()




def collect(output_dir: Path, delay: float = 1.0, bvids: list[str] | None = None) -> dict[str, Any]:
    bvids = BVIDS if bvids is None else bvids
    raw_dir = output_dir / "raw"
    subtitle_dir = output_dir / "subtitles"
    records: list[dict[str, Any]] = []
    for index, bvid in enumerate(bvids, start=1):
        record: dict[str, Any] = {"index": index, "bvid": bvid}
        try:
            view_url, view = _request_json("/x/web-interface/view", {"bvid": bvid})
            view_path = raw_dir / f"{index:02d}_{bvid}_view.json"
            _save_json(view_path, view)
            record["view_raw_path"] = str(view_path.relative_to(output_dir)).replace("\\", "/")
            record["view_raw_sha256"] = file_sha256(view_path)
            record["view_url"] = view_url
            record["view_code"] = view.get("code")
            data = view.get("data") or {}
            owner = data.get("owner") or {}
            record.update({
                "title": data.get("title"),
                "pubdate": data.get("pubdate"),
                "pubdate_iso": datetime.fromtimestamp(data["pubdate"], tz=timezone.utc).isoformat()
                if isinstance(data.get("pubdate"), (int, float)) else None,
                "ctime": data.get("ctime"),
                "desc": data.get("desc"),
                "cid": data.get("cid"),
                "duration": data.get("duration"),
                "owner_uid": owner.get("mid"),
                "owner_name": owner.get("name"),
                "pages": [
                    {"page": p.get("page"), "cid": p.get("cid"), "part": p.get("part"),
                     "duration": p.get("duration"), "pubdate": p.get("pubdate")}
                    for p in (data.get("pages") or [])
                ],
            })
            if record["owner_uid"] != OWNER_UID:
                record["owner_uid_mismatch"] = True
            cid = record.get("cid")
            if cid:
                time.sleep(delay)
                player_url, player = _request_json("/x/player/v2", {"bvid": bvid, "cid": cid})
                player_path = raw_dir / f"{index:02d}_{bvid}_player_v2.json"
                _save_json(player_path, player)
                record["player_raw_path"] = str(player_path.relative_to(output_dir)).replace("\\", "/")
                record["player_raw_sha256"] = file_sha256(player_path)
                record["player_url"] = player_url
                record["player_code"] = player.get("code")
                subtitles = ((player.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
                record["subtitle_count"] = len(subtitles)
                record["subtitles"] = []
                for sub_index, subtitle in enumerate(subtitles, start=1):
                    sub_url = subtitle.get("subtitle_url") or subtitle.get("url")
                    sub_record = {"id": subtitle.get("id"), "lan": subtitle.get("lan"),
                                  "lan_doc": subtitle.get("lan_doc"), "subtitle_url": sub_url}
                    if sub_url:
                        if sub_url.startswith("//"):
                            sub_url = "https:" + sub_url
                        try:
                            request = Request(sub_url, headers=HEADERS)
                            with urlopen(request, timeout=30) as response:
                                payload = json.loads(response.read().decode("utf-8"))
                            sub_path = subtitle_dir / f"{index:02d}_{bvid}_{sub_index:02d}.json"
                            _save_json(sub_path, payload)
                            sub_record["saved_path"] = str(sub_path.relative_to(output_dir)).replace("\\", "/")
                        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                            sub_record["download_error"] = _error_record(exc)
                    record["subtitles"].append(sub_record)
                time.sleep(delay)
                playurl_url, playurl = _request_json(
                    "/x/player/playurl", {"bvid": bvid, "cid": cid, "qn": 16, "fnval": 0}
                )
                playurl_path = raw_dir / f"{index:02d}_{bvid}_playurl.json"
                _save_json(playurl_path, playurl)
                record["playurl_raw_path"] = str(playurl_path.relative_to(output_dir)).replace("\\", "/")
                record["playurl_raw_sha256"] = file_sha256(playurl_path)
                record["playurl_url"] = playurl_url
                record["playurl_code"] = playurl.get("code")
                play_data = playurl.get("data") or {}
                durls = play_data.get("durl") or []
                record["media"] = []
                for media_index, segment in enumerate(durls, start=1):
                    media_url = segment.get("url")
                    media_record = {"order": segment.get("order"), "length": segment.get("length"),
                                    "size": segment.get("size"), "url": media_url}
                    if media_url:
                        media_path = output_dir / "media" / f"{index:02d}_{bvid}_{media_index:02d}.mp4"
                        try:
                            media_record["saved_path"] = str(media_path.relative_to(output_dir)).replace("\\", "/")
                            candidates = [media_url, *(segment.get("backup_url") or [])]
                            download_error = None
                            for candidate_index, candidate in enumerate(candidates):
                                try:
                                    media_record["sha256"] = _download(candidate, media_path, bvid)
                                    media_record["bytes"] = media_path.stat().st_size
                                    media_record["download_source"] = "primary" if candidate_index == 0 else f"backup_{candidate_index}"
                                    break
                                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                                    download_error = _error_record(exc)
                                    if media_path.exists():
                                        media_path.unlink()
                            else:
                                media_record["download_error"] = download_error
                        except (HTTPError, URLError, TimeoutError, OSError) as exc:
                            media_record["download_error"] = _error_record(exc)
                    record["media"].append(media_record)
            else:
                record["subtitle_count"] = None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            record["error"] = _error_record(exc)
        records.append(record)
        _save_json(output_dir / "manifest.json", {"owner_uid": OWNER_UID, "videos": records})
        if index < len(bvids):
            time.sleep(delay)
    summary = {"owner_uid": OWNER_UID, "requested_count": len(bvids), "completed_count": len(records),
               "owner_mismatch_count": sum(r.get("owner_uid") not in (None, OWNER_UID) for r in records),
               "error_count": sum("error" in r for r in records),
               "subtitle_video_count": sum((r.get("subtitle_count") or 0) > 0 for r in records),
               "videos": records}
    _save_json(output_dir / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--bvid", action="append", help="Explicit videos; omitted uses the archived twenty-video list")
    args = parser.parse_args()
    summary = collect(args.output, delay=args.delay, bvids=args.bvid)
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

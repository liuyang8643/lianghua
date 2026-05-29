"""检查飞书 audit jsonl 文件的内容（飞书发送审计工具）。

用法:
    python scripts/inspect_lark_audit.py [YYYYMMDD]
    python scripts/inspect_lark_audit.py 20260528 --idx 0       # 看某一条
    python scripts/inspect_lark_audit.py 20260528 --idx 0 --raw # 看 raw json
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def _summary(records: list[dict]) -> None:
    print(f'共 {len(records)} 条 audit 记录')
    print('-' * 80)
    for i, r in enumerate(records):
        title = r.get('title') or r.get('content', '')[:30] or r.get('file', {}).get('name', '')
        template = r.get('template', '-')
        ok = '✓' if r.get('ok') else '✗'
        method = r.get('method', '-')
        ts = r.get('ts', '')[-12:]
        print(f'  [{i:2d}] {ts} {ok} {method:20s} template={template:10s} title={title}')


def _detail(rec: dict) -> None:
    print(f"==== {rec.get('title') or rec.get('content', '')!r} ====")
    print(f"  ts: {rec.get('ts')}")
    print(f"  method: {rec.get('method')}")
    print(f"  ok: {rec.get('ok')}, response: {rec.get('response')}")
    card = rec.get('card')
    if card:
        print(f"  schema: {card.get('schema')}")
        header = card.get('header', {})
        print(f"  template: {header.get('template')}")
        title = header.get('title', {}).get('content')
        if title:
            print(f"  title: {title}")
        subtitle = header.get('subtitle', {}).get('content')
        if subtitle:
            print(f"  subtitle: {subtitle}")
        elements = card.get('body', {}).get('elements', [])
        print(f"  elements: {len(elements)}")
        for j, e in enumerate(elements):
            tag = e.get('tag')
            if tag == 'div':
                content = e['text']['content']
                short = content if len(content) <= 120 else content[:120] + '...'
                print(f"    [{j:2d}] div: {short!r}")
            elif tag == 'hr':
                print(f"    [{j:2d}] hr")
            elif tag == 'table':
                cols = e.get('columns', [])
                rows = e.get('rows', [])
                print(f"    [{j:2d}] table[id={e.get('element_id')}]: "
                      f"{len(cols)} cols × {len(rows)} rows, "
                      f"freeze_first={e.get('freeze_first_column')}, "
                      f"page_size={e.get('page_size')}")
                print(f"          columns: {[c.get('display_name') for c in cols]}")
                if rows:
                    print(f"          row[0]: {rows[0]}")
                    if len(rows) > 1:
                        print(f"          row[-1]: {rows[-1]}")
            else:
                print(f"    [{j:2d}] {tag}")
    if rec.get('file'):
        print(f"  file: {rec['file']}")
    if rec.get('error'):
        print(f"  error: {rec['error']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date', nargs='?', default=datetime.now().strftime('%Y%m%d'),
                    help='YYYYMMDD（默认今天）')
    ap.add_argument('--idx', type=int, default=None, help='详看第 N 条')
    ap.add_argument('--raw', action='store_true', help='输出 raw JSON')
    args = ap.parse_args()

    audit_dir = Path(__file__).resolve().parents[1] / 'data' / 'live_trades' / 'lark_audit'
    path = audit_dir / f'{args.date}.jsonl'
    records = _load(path)

    if args.idx is None:
        _summary(records)
        return
    if not 0 <= args.idx < len(records):
        raise SystemExit(f'idx 越界: {args.idx} not in [0, {len(records)})')
    rec = records[args.idx]
    if args.raw:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    else:
        _detail(rec)


if __name__ == '__main__':
    main()

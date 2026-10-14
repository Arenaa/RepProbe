#!/usr/bin/env python3
"""Merge eval JSONL shards. Drops truncated lines. Last non-error row wins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ok(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def key_fn(row: dict, mode: str):
    if mode == "vid":
        return row.get("video_id")
    return (row.get("video_id"), row.get("exp"), row.get("mode"))


def merge(paths: list[Path], out: Path, mode: str) -> int:
    by: dict = {}
    for p in paths:
        for r in load_ok(p):
            k = key_fn(r, mode)
            if not k or (isinstance(k, tuple) and any(x is None for x in k)):
                continue
            old = by.get(k)
            if old is None or (old.get("error") and not r.get("error")):
                by[k] = r
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v in by.values():
            f.write(json.dumps(v) + "\n")
    return len(by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("vid", "order"), default="vid")
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    n = merge([Path(p) for p in args.inputs], Path(args.out), args.mode)
    print(f"merged {n} -> {args.out}")


if __name__ == "__main__":
    main()

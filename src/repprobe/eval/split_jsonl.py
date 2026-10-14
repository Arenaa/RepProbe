#!/usr/bin/env python3
"""Split a JSONL file into N round-robin shards (same scheme as eval_nrep --shard i/n)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_rows(rows: list[dict], shard: int, n_shards: int) -> list[dict]:
    return [r for j, r in enumerate(rows) if j % n_shards == shard]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--out-prefix", type=Path, required=True, help="writes {prefix}{i}.jsonl")
    args = ap.parse_args()
    rows = load_jsonl(args.inp)
    n = args.shards
    if n < 1:
        raise SystemExit("--shards must be >= 1")
    sizes = []
    for i in range(n):
        part = split_rows(rows, i, n)
        out = Path(f"{args.out_prefix}{i}.jsonl")
        write_jsonl(out, part)
        sizes.append(len(part))
    print(json.dumps({"input": str(args.inp), "total": len(rows), "shards": n, "sizes": sizes}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score n-rep prediction JSONLs overall, per k, and few/mid/many bands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BANDS = {
    "few k=1-6": range(1, 7),
    "mid k=7-13": range(7, 14),
    "many k=14-20": range(14, 21),
}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    # unique by video_id (LongVILA shards)
    by = {}
    for r in rows:
        by[r["video_id"]] = r
    return list(by.values())


def mae_exact(rows: list[dict]) -> tuple[float, float, int, int]:
    parsed = [r for r in rows if r.get("pred") is not None and r.get("gt") is not None]
    if not parsed:
        return float("nan"), float("nan"), 0, len(rows)
    mae = sum(abs(r["pred"] - r["gt"]) for r in parsed) / len(parsed)
    exact = sum(r["pred"] == r["gt"] for r in parsed) / len(parsed)
    return mae, exact, len(parsed), len(rows) - len(parsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="prediction JSONL files")
    args = ap.parse_args()
    print(f"{'model':28s} {'overall':>14s} {'few 1-6':>14s} {'mid 7-13':>14s} {'many 14-20':>14s} best")
    for p in args.preds:
        path = Path(p)
        rows = load_rows(path)
        name = path.name.replace("eval10000__", "").replace("eval2000__", "").replace(".jsonl", "")
        om, oe, ok, fail = mae_exact(rows)
        parts = []
        best = None
        best_mae = 1e9
        for b, ks in BANDS.items():
            sub = [r for r in rows if r.get("gt") is not None and int(r["gt"]) in ks]
            mae, ex, n, _ = mae_exact(sub)
            parts.append(f"{mae:5.2f}/{ex:4.0%} n={n}")
            if mae < best_mae:
                best_mae, best = mae, b.split()[0]
        print(
            f"{name[:28]:28s} {om:5.2f}/{oe:4.0%}n={ok}  "
            + "  ".join(parts)
            + f"  {best}  fail={fail}"
        )
        print("  per-k MAE:", end="")
        for k in range(1, 21):
            sub = [r for r in rows if r.get("gt") == k and r.get("pred") is not None]
            if not sub:
                print(f" {k}:-", end="")
                continue
            mae = sum(abs(r["pred"] - r["gt"]) for r in sub) / len(sub)
            print(f" {k}:{mae:.2f}", end="")
        print()


if __name__ == "__main__":
    main()

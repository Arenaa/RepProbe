#!/usr/bin/env python3
"""Build a 10,000-clip eval set: 500 per count k=1–20 from all.jsonl.

Nests eval_2000 (and thus smoke_200). Spreads source, then position, then action.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from repprobe.paths import REPCOUNT

ROOT = REPCOUNT
PER_K = 500
SEED = 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def fill_k(keep: list[dict], pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    have = {r["video_id"] for r in keep}
    remaining = [r for r in pool if r["video_id"] not in have]
    rng.shuffle(remaining)
    selected = list(keep)
    pos = Counter(r["position"] for r in selected)
    src = Counter(r["source_video"] for r in selected)
    act = Counter(r["action_type"] for r in selected)

    def cand_cost(cand: dict) -> tuple:
        p = pos.copy()
        p[cand["position"]] += 1
        a = act.copy()
        a[cand["action_type"]] += 1
        return (
            src[cand["source_video"]] + 1,
            max(p.values()) - min(p.values()),
            max(a.values()) - min(a.values()),
            cand["video_id"],
        )

    while len(selected) < n and remaining:
        best_i = min(range(len(remaining)), key=lambda i: cand_cost(remaining[i]))
        cand = remaining.pop(best_i)
        selected.append(cand)
        pos[cand["position"]] += 1
        src[cand["source_video"]] += 1
        act[cand["action_type"]] += 1
    return selected


def main():
    pool_all = load_jsonl(ROOT / "all.jsonl")
    keep_all = load_jsonl(ROOT / "eval_2000.jsonl")
    rng = random.Random(SEED)
    by_k_pool: dict[int, list[dict]] = defaultdict(list)
    by_k_keep: dict[int, list[dict]] = defaultdict(list)
    for r in pool_all:
        by_k_pool[int(r["count"])].append(r)
    for r in keep_all:
        by_k_keep[int(r["count"])].append(r)

    chosen = []
    for k in range(1, 21):
        pool = by_k_pool[k]
        if len(pool) < PER_K:
            raise SystemExit(f"k={k} only {len(pool)} clips, need {PER_K}")
        rows = fill_k(by_k_keep.get(k, []), pool, PER_K, rng)
        if len(rows) != PER_K:
            raise SystemExit(f"k={k} got {len(rows)}")
        chosen.extend(rows)

    out = ROOT / "eval_10000.jsonl"
    with out.open("w") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eval_ids = {r["video_id"] for r in chosen}
    print(f"wrote {len(chosen)} -> {out}")
    print("nested eval_2000", sum(1 for r in keep_all if r["video_id"] in eval_ids))
    print("by k", dict(Counter(r["count"] for r in chosen)))
    print("pos", dict(Counter(r["position"] for r in chosen)))
    print("split", dict(Counter(r["source_split"] for r in chosen)))
    print("sources", len(set(r["source_video"] for r in chosen)))
    leftover = Counter()
    for r in load_jsonl(ROOT / "train.jsonl"):
        if r["video_id"] not in eval_ids:
            leftover[int(r["count"])] += 1
    print("train leftover for probe", dict(sorted(leftover.items())))
    print("min leftover", min(leftover[k] for k in range(1, 21)))
    for k in range(1, 21):
        sub = [r for r in chosen if r["count"] == k]
        print(
            f"k={k:2d} n={len(sub):3d} pos={dict(Counter(r['position'] for r in sub))} "
            f"src={len(set(r['source_video'] for r in sub))} "
            f"split={dict(Counter(r['source_split'] for r in sub))} "
            f"max_src={max(Counter(r['source_video'] for r in sub).values())}"
        )


if __name__ == "__main__":
    main()

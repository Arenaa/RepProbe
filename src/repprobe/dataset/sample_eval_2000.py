#!/usr/bin/env python3
"""Build a 2000-clip eval set: 100 per count k=1..20 from test.

Keeps the original smoke_200 clips, then fills each class by spreading
source videos, then window position, then action type.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from repprobe.paths import REPCOUNT

ROOT = REPCOUNT
PER_K = 100
SEED = 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def cost(selected: list[dict], cand: dict) -> tuple:
    pos = Counter(r["position"] for r in selected)
    src = Counter(r["source_video"] for r in selected)
    act = Counter(r["action_type"] for r in selected)
    pos[cand["position"]] += 1
    src[cand["source_video"]] += 1
    act[cand["action_type"]] += 1
    pos_imbal = max(pos.values()) - min(pos.values())
    act_imbal = max(act.values()) - min(act.values())
    return (src[cand["source_video"]], pos_imbal, act_imbal, cand["video_id"])


def fill_k(keep: list[dict], pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    have = {r["video_id"] for r in keep}
    remaining = [r for r in pool if r["video_id"] not in have]
    rng.shuffle(remaining)
    selected = list(keep)
    while len(selected) < n and remaining:
        best_i = min(range(len(remaining)), key=lambda i: cost(selected, remaining[i]))
        selected.append(remaining.pop(best_i))
    return selected


def main():
    test = load_jsonl(ROOT / "test.jsonl")
    smoke = load_jsonl(ROOT / "smoke_200.jsonl")
    rng = random.Random(SEED)
    by_k_test: dict[int, list[dict]] = defaultdict(list)
    by_k_smoke: dict[int, list[dict]] = defaultdict(list)
    for r in test:
        by_k_test[int(r["count"])].append(r)
    for r in smoke:
        by_k_smoke[int(r["count"])].append(r)

    chosen = []
    for k in range(1, 21):
        pool = by_k_test[k]
        if len(pool) < PER_K:
            raise SystemExit(f"k={k} only {len(pool)} test clips, need {PER_K}")
        keep = by_k_smoke.get(k, [])
        rows = fill_k(keep, pool, PER_K, rng)
        if len(rows) != PER_K:
            raise SystemExit(f"k={k} got {len(rows)}")
        chosen.extend(rows)

    out = ROOT / "eval_2000.jsonl"
    with out.open("w") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(chosen)} -> {out}")
    print("nested smoke", sum(1 for r in chosen if r["video_id"] in {s["video_id"] for s in smoke}))
    print("by k", dict(Counter(r["count"] for r in chosen)))
    print("pos", dict(Counter(r["position"] for r in chosen)))
    print("actions", len(set(r["action_type"] for r in chosen)), dict(Counter(r["action_type"] for r in chosen)))
    print("sources", len(set(r["source_video"] for r in chosen)))
    for k in range(1, 21):
        sub = [r for r in chosen if r["count"] == k]
        print(
            f"k={k:2d} n={len(sub):3d} pos={dict(Counter(r['position'] for r in sub))} "
            f"src={len(set(r['source_video'] for r in sub))} "
            f"act={len(set(r['action_type'] for r in sub))} "
            f"max_src={max(Counter(r['source_video'] for r in sub).values())}"
        )


if __name__ == "__main__":
    main()

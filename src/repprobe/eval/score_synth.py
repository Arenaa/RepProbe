#!/usr/bin/env python3
"""Score synthetic-10k counting predictions (generalists + specialists).

Reads data/synth_reps_10k/preds/synth10k__*.jsonl, pairs preds to the manifest
ground truth, and reports MAE / OBO / exact overall, by band (few/mid/many),
and per pattern family. LongVILA shards are merged automatically.

    python3 score_synth10k.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from repprobe.paths import SYNTH

PRED = SYNTH / "preds"
MANIFEST = SYNTH / "eval_synthetic.jsonl"
CLIP_LO, CLIP_HI = 0, 40
BANDS = (("few", 1, 6), ("mid", 7, 13), ("many", 14, 20))

MODELS = [
    ("Cosmos-Reason2-8B", "synth10k__nvidia_Cosmos-Reason2-8B.jsonl"),
    ("Qwen3-VL-8B", "synth10k__Qwen_Qwen3-VL-8B-Instruct.jsonl"),
    ("InternVL3.5-8B", "synth10k__OpenGVLab_InternVL3_5-8B-HF.jsonl"),
    ("VideoChat-R1.5-7B", "synth10k__OpenGVLab_VideoChat-R1_5-7B.jsonl"),
    ("Video-R1-7B", "synth10k__Video-R1_Video-R1-7B.jsonl"),
    ("LongVILA-R1-7B", "synth10k__Efficient-Large-Model_LongVILA-R1-7B*.jsonl"),
    ("LLaVA-OV-7B", "synth10k__llava-hf_llava-onevision-qwen2-7b-ov-hf.jsonl"),
    ("RepNet (specialist)", "synth10k__google_RepNet.jsonl"),
    ("ESCounts (specialist)", "synth10k__EveryShotCounts_ESCounts-RepCount.jsonl"),
]


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def load_preds(pattern: str) -> dict[str, dict]:
    d: dict[str, dict] = {}
    for fp in sorted(glob.glob(str(PRED / pattern))):
        for r in load_jsonl(Path(fp)):
            vid = r.get("video_id")
            if vid is not None:
                d[vid] = r  # later shards overwrite dup ids harmlessly
    return d


def metrics(pairs):
    n = len(pairs)
    if n == 0:
        return None
    mae = sum(abs(p - g) for p, g in pairs) / n
    obo = sum(abs(p - g) <= 1 for p, g in pairs) / n
    ex = sum(p == g for p, g in pairs) / n
    return dict(n=n, mae=mae, obo=obo, exact=ex)


def main():
    gt = {r["video_id"]: r for r in load_jsonl(MANIFEST)}
    total = len(gt)
    summary = {}
    for name, pattern in MODELS:
        preds = load_preds(pattern)
        if not preds:
            summary[name] = {"status": "no preds yet"}
            continue
        all_pairs, band_pairs, fam_pairs = [], {b: [] for b, _, _ in BANDS}, defaultdict(list)
        n_parse = 0
        for vid, g in gt.items():
            r = preds.get(vid)
            if r is None:
                continue
            p = r.get("pred")
            if p is None:
                continue
            n_parse += 1
            k = int(g["count"])
            p = max(CLIP_LO, min(CLIP_HI, int(p)))
            all_pairs.append((p, k))
            for b, lo, hi in BANDS:
                if lo <= k <= hi:
                    band_pairs[b].append((p, k))
            fam_pairs[g.get("family", "?")].append((p, k))
        summary[name] = {
            "n_seen": len(preds),
            "n_parseable": n_parse,
            "parse_rate": round(n_parse / total, 4),
            "overall": metrics(all_pairs),
            "bands": {b: metrics(band_pairs[b]) for b, _, _ in BANDS},
            "by_family": {f: metrics(v) for f, v in sorted(fam_pairs.items())},
        }
    (PRED / "synth10k_behavior_scores.json").write_text(json.dumps(summary, indent=2) + "\n")

    # pretty table
    print(f"\nSynthetic-10k counting  ({total} clips, k=1-20, 10 families)\n")
    hdr = f"{'model':<24} {'n':>6} {'MAE':>7} {'OBO%':>6} {'exact%':>7} | {'few':>6} {'mid':>6} {'many':>6}"
    print(hdr); print("-" * len(hdr))
    for name, _ in MODELS:
        s = summary[name]
        if "overall" not in s or s["overall"] is None:
            print(f"{name:<24} {'--- (no preds yet)':>40}")
            continue
        o = s["overall"]; b = s["bands"]
        def bm(x): return f"{x['mae']:.2f}" if x else "  -"
        print(f"{name:<24} {o['n']:>6} {o['mae']:>7.2f} {100*o['obo']:>6.1f} {100*o['exact']:>7.1f} | "
              f"{bm(b['few']):>6} {bm(b['mid']):>6} {bm(b['many']):>6}")
    print(f"\nwrote {PRED/'synth10k_behavior_scores.json'}")


if __name__ == "__main__":
    main()

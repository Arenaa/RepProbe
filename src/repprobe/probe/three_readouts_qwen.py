#!/usr/bin/env python3
"""Qwen three-readout table on the balanced 10k set, few/mid/many bands.

Mirrors score_three_readouts.py but for Qwen and with all four bands.
  generation  = eval10000__Qwen_Qwen3-VL-8B-Instruct.jsonl (nframes=16)
  head_argmax = eval10000__qwen_number_prior.jsonl (argmax_int, nframes=16)
  l35_probe   = eval10000__qwen_hidden_probe.jsonl (L35 last, 1k labels, nframes=16)
"""
from __future__ import annotations

import json
from pathlib import Path

from repprobe.paths import REPCOUNT

PRED = REPCOUNT / "preds"


def load_map(path: Path, key="pred") -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        vid = r.get("video_id")
        if not vid:
            continue
        if vid not in out or (out[vid].get(key) is None and r.get(key) is not None):
            out[vid] = r
    return out


def band(pairs, lo, hi):
    s = [(p, g) for p, g in pairs if lo <= g <= hi]
    n = len(s)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mae": sum(abs(p - g) for p, g in s) / n,
        "obo": sum(abs(p - g) <= 1 for p, g in s) / n,
        "exact": sum(p == g for p, g in s) / n,
    }


def main() -> None:
    gen = load_map(PRED / "eval10000__Qwen_Qwen3-VL-8B-Instruct.jsonl")
    prior = load_map(PRED / "eval10000__qwen_number_prior.jsonl", key="argmax_int")
    probe = load_map(PRED / "eval10000__qwen_hidden_probe.jsonl")

    ids = set(gen) & set(prior) & set(probe)
    rows = {"generation": [], "head_argmax": [], "l35_probe": []}
    for vid in ids:
        g = gen[vid].get("pred")
        a = prior[vid].get("argmax_int") or prior[vid].get("pred")
        p = probe[vid].get("pred")
        gt = probe[vid].get("gt") or gen[vid].get("gt")
        if None in (g, a, p, gt):
            continue
        gt = int(gt)
        rows["generation"].append((int(g), gt))
        rows["head_argmax"].append((int(a), gt))
        rows["l35_probe"].append((int(p), gt))

    table = {"n_paired": len(rows["generation"]), "readouts": {}}
    for name, pairs in rows.items():
        n = len(pairs)
        table["readouts"][name] = {
            "n": n,
            "mae": sum(abs(p - g) for p, g in pairs) / n,
            "obo": sum(abs(p - g) <= 1 for p, g in pairs) / n,
            "exact": sum(p == g for p, g in pairs) / n,
            "few": band(pairs, 1, 6),
            "mid": band(pairs, 7, 13),
            "many": band(pairs, 14, 20),
        }

    out = PRED / "three_readouts_qwen.json"
    out.write_text(json.dumps(table, indent=2) + "\n")

    labels = {
        "head_argmax": "Integer language head",
        "generation": "Generation",
        "l35_probe": "Count probe (L35)",
    }
    print(f"=== Qwen 10k three readouts (paired n={table['n_paired']}) ===")
    hdr = f"{'readout':<22} {'MAE':>6} {'OBO':>6} | {'few MAE':>7} {'few OBO':>7} | {'mid MAE':>7} {'mid OBO':>7} | {'many MAE':>8} {'many OBO':>8}"
    print(hdr)
    for name in ("head_argmax", "generation", "l35_probe"):
        s = table["readouts"][name]
        f, m, y = s["few"], s["mid"], s["many"]
        print(
            f"{labels[name]:<22} {s['mae']:6.2f} {100*s['obo']:6.1f} | "
            f"{f['mae']:7.2f} {100*f['obo']:7.1f} | "
            f"{m['mae']:7.2f} {100*m['obo']:7.1f} | "
            f"{y['mae']:8.2f} {100*y['obo']:8.1f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

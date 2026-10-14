#!/usr/bin/env python3
"""Fairness table: generation vs unsupervised head-argmax vs supervised L35 probe.

Uses existing JSONLs only (CPU). Also dumps OVR many-band raw pred histograms.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from repprobe.paths import REPO_ROOT, REPCOUNT

PRED = REPCOUNT / "preds"
OVR = REPO_ROOT / "data" / "ovr" / "preds"


def load_map(path: Path, pred_key="pred") -> dict[str, dict]:
    out = {}
    if not path.is_file():
        return out
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        vid = r.get("video_id")
        if not vid:
            continue
        if vid not in out or (out[vid].get(pred_key) is None and r.get(pred_key) is not None):
            out[vid] = r
    return out


def score(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    mae = sum(abs(p - g) for p, g in pairs) / n
    exact = sum(p == g for p, g in pairs) / n
    obo = sum(abs(p - g) <= 1 for p, g in pairs) / n
    ge14 = sum(p >= 14 for p, _g in pairs) / n
    le10 = sum(p <= 10 for p, _g in pairs) / n
    many = [(p, g) for p, g in pairs if g >= 14]
    many_s = None
    if many:
        many_s = {
            "n": len(many),
            "mae": sum(abs(p - g) for p, g in many) / len(many),
            "obo": sum(abs(p - g) <= 1 for p, g in many) / len(many),
            "pred_ge14": sum(p >= 14 for p, _g in many) / len(many),
            "mean_pred": sum(p for p, _g in many) / len(many),
        }
    return {
        "n": n,
        "mae": mae,
        "exact": exact,
        "obo": obo,
        "pred_ge14": ge14,
        "pred_le10": le10,
        "many": many_s,
    }


def main() -> None:
    gen = load_map(PRED / "eval10000__nvidia_Cosmos-Reason2-8B.jsonl")
    prior = load_map(PRED / "eval10000__number_prior.jsonl", pred_key="argmax_int")
    probe = load_map(PRED / "eval10000__cosmos_hidden_probe.jsonl")

    ids = set(gen) & set(prior) & set(probe)
    rows = {
        "generation": [],
        "head_argmax": [],
        "l35_probe": [],
    }
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

    table = {name: score(pairs) for name, pairs in rows.items()}
    table["n_paired"] = len(rows["generation"])
    table["note"] = (
        "head_argmax = unsupervised integer-set softmax at last prompt token "
        "(eval_number_prior). l35_probe = supervised logistic on residual "
        "(1000 labeled train clips). generation = parsed decode."
    )
    out = PRED / "three_readouts.json"
    out.write_text(json.dumps(table, indent=2) + "\n")
    print("=== Cosmos 10k: three readouts (paired) ===")
    print(f"{'readout':<16} {'n':>5} {'MAE':>7} {'exact':>7} {'OBO':>7} {'many MAE':>9} {'many OBO':>9}")
    for name in ("generation", "head_argmax", "l35_probe"):
        s = table[name]
        m = s.get("many") or {}
        print(
            f"{name:<16} {s['n']:5d} {s['mae']:7.3f} {100*s['exact']:6.1f}% {100*s['obo']:6.1f}% "
            f"{m.get('mae', float('nan')):9.2f} {100*m.get('obo', float('nan')):8.1f}%"
        )
    print(f"wrote {out}")

    ovr_eval = REPO_ROOT / "data/ovr/eval_ovr_1000.jsonl"
    if not ovr_eval.is_file():
        return
    ev = [json.loads(l) for l in ovr_eval.open() if l.strip()]
    many_ids = {r["video_id"]: int(r["count"]) for r in ev if int(r["count"]) >= 14}
    models = {
        "Qwen3-VL-8B": OVR / "ovr1000__Qwen_Qwen3-VL-8B-Instruct.jsonl",
        "Cosmos-Reason2-8B": OVR / "ovr1000__nvidia_Cosmos-Reason2-8B.jsonl",
        "Video-R1-7B": OVR / "ovr1000__Video-R1_Video-R1-7B.jsonl",
        "InternVL3.5-8B": OVR / "ovr1000__OpenGVLab_InternVL3_5-8B-HF.jsonl",
        "VideoChat-R1.5-7B": OVR / "ovr1000__OpenGVLab_VideoChat-R1_5-7B.jsonl",
        "LLaVA-OneVision-7B": OVR / "ovr1000__llava-hf_llava-onevision-qwen2-7b-ov-hf.jsonl",
        "ESCounts": OVR / "ovr1000__EveryShotCounts_ESCounts-RepCount.jsonl",
        "RepNet": OVR / "ovr1000__google_RepNet.jsonl",
    }
    dump = {"n_many": len(many_ids), "k_hist": dict(Counter(many_ids.values())), "models": {}}
    print("\n=== OVR many (k>=14) raw pred hist ===")
    for name, path in models.items():
        m = load_map(path)
        preds = []
        fail = 0
        for vid, gt in many_ids.items():
            r = m.get(vid)
            if not r or r.get("pred") is None:
                fail += 1
                continue
            preds.append(int(r["pred"]))
        hist = dict(Counter(preds).most_common())
        dump["models"][name] = {
            "n": len(preds),
            "fail": fail,
            "pred_hist": hist,
            "n_pred_ge14": sum(p >= 14 for p in preds),
            "max_pred": max(preds) if preds else None,
            "mean_pred": sum(preds) / len(preds) if preds else None,
            "obo": sum(abs(p - many_ids[vid]) <= 1 for vid, p in zip(
                [v for v, gt in many_ids.items() if m.get(v) and m[v].get("pred") is not None],
                preds,
            )) / len(preds) if preds else None,
        }
        # fix obo properly
        errs = []
        for vid, gt in many_ids.items():
            r = m.get(vid)
            if r and r.get("pred") is not None:
                errs.append(abs(int(r["pred"]) - gt) <= 1)
        dump["models"][name]["obo"] = sum(errs) / len(errs) if errs else None
        print(
            f"{name:<22} n={len(preds):3d} fail={fail:2d} max={dump['models'][name]['max_pred']} "
            f"ge14={dump['models'][name]['n_pred_ge14']} mean={dump['models'][name]['mean_pred']}"
        )
    ovr_out = OVR / "ovr1000_many_raw.json"
    ovr_out.write_text(json.dumps(dump, indent=2) + "\n")
    print(f"wrote {ovr_out}")


if __name__ == "__main__":
    main()

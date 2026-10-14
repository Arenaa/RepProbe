#!/usr/bin/env python3
"""Within-synth linear probe (train on synth, test on synth) via 5-fold CV.

Reads last-layer hidden states cached by eval_cosmos_probe.py (--layers last)
for Cosmos and Qwen on the synthetic-10k set, fits a StandardScaler +
LogisticRegression probe with StratifiedKFold(5) on k, and reports
MAE / OBO / exact overall, by band, and per family. Predictions and metrics
are written next to the caches.

    python3 score_synth_within_cv.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from repprobe.paths import SYNTH

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PRED = SYNTH / "preds"
MANIFEST = SYNTH / "eval_synthetic.jsonl"
BANDS = (("few", 1, 6), ("mid", 7, 13), ("many", 14, 20))

CACHES = [
    ("Cosmos-Reason2-8B", "synth10k__cosmos_hidden_probe.emb.npz"),
    ("Qwen3-VL-8B", "synth10k__qwen_hidden_probe.emb.npz"),
]


def load_last_feature(npz_path: Path):
    blob = np.load(npz_path, allow_pickle=True)
    ids = [str(v) for v in blob["ids"]]
    last_keys = [k for k in blob.files if k.endswith("::last")]
    if not last_keys:
        raise SystemExit(f"no ::last feature in {npz_path}; keys={blob.files}")
    # highest layer index if several
    def ix(k):
        import re
        m = re.search(r"layers\.(\d+)", k)
        return int(m.group(1)) if m else -1
    key = sorted(last_keys, key=ix)[-1]
    X = np.asarray(blob[key])
    return ids, X, key


def band_metrics(pred, y, lo, hi):
    m = (y >= lo) & (y <= hi)
    if not m.any():
        return None
    return dict(n=int(m.sum()),
                mae=float(np.mean(np.abs(pred[m] - y[m]))),
                obo=float(np.mean(np.abs(pred[m] - y[m]) <= 1)),
                exact=float(np.mean(pred[m] == y[m])))


def main():
    gt = {r["video_id"]: r for r in (json.loads(l) for l in MANIFEST.open() if l.strip())}
    all_metrics = {}
    for name, fname in CACHES:
        path = PRED / fname
        if not path.is_file():
            all_metrics[name] = {"status": "cache missing"}
            print(f"[skip] {name}: {path} not found")
            continue
        ids, X, key = load_last_feature(path)
        keep = [i for i, v in enumerate(ids) if v in gt]
        ids = [ids[i] for i in keep]
        X = X[keep]
        y = np.array([int(gt[v]["count"]) for v in ids], dtype=np.int64)
        fam = [gt[v].get("family", "?") for v in ids]
        print(f"[{name}] feat={key}  n={len(y)}  dim={X.shape[1]}")

        pred = np.zeros_like(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))

        overall = dict(n=int(len(y)),
                       mae=float(np.mean(np.abs(pred - y))),
                       obo=float(np.mean(np.abs(pred - y) <= 1)),
                       exact=float(np.mean(pred == y)),
                       pearson=float(np.corrcoef(pred, y)[0, 1]))
        by_fam = {}
        fam_idx = defaultdict(list)
        for i, f in enumerate(fam):
            fam_idx[f].append(i)
        for f, idx in sorted(fam_idx.items()):
            idx = np.array(idx)
            by_fam[f] = dict(n=len(idx), mae=float(np.mean(np.abs(pred[idx] - y[idx]))),
                             obo=float(np.mean(np.abs(pred[idx] - y[idx]) <= 1)))
        all_metrics[name] = {
            "feat": key, "cv": "StratifiedKFold(5)",
            "overall": overall,
            "bands": {b: band_metrics(pred, y, lo, hi) for b, lo, hi in BANDS},
            "by_family": by_fam,
        }
        # save per-clip preds
        with (PRED / f"synth10k_within_cv_preds__{name.split('-')[0].lower()}.jsonl").open("w") as fo:
            for v, p, g in zip(ids, pred.tolist(), y.tolist()):
                fo.write(json.dumps({"video_id": v, "pred": int(p), "gt": int(g),
                                     "family": gt[v].get("family")}) + "\n")

    (PRED / "synth10k_within_cv_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")

    print("\nWithin-synth last-layer probe (5-fold CV)\n")
    hdr = f"{'model':<20} {'n':>6} {'MAE':>7} {'OBO%':>6} {'exact%':>7} {'r':>6} | {'few':>6} {'mid':>6} {'many':>6}"
    print(hdr); print("-" * len(hdr))
    for name, _ in CACHES:
        s = all_metrics.get(name, {})
        if "overall" not in s:
            print(f"{name:<20} {'--- (' + s.get('status','pending') + ')':>40}"); continue
        o = s["overall"]; b = s["bands"]
        def bm(x): return f"{x['mae']:.2f}" if x else "  -"
        print(f"{name:<20} {o['n']:>6} {o['mae']:>7.2f} {100*o['obo']:>6.1f} {100*o['exact']:>7.1f} {o['pearson']:>6.3f} | "
              f"{bm(b['few']):>6} {bm(b['mid']):>6} {bm(b['many']):>6}")
    print(f"\nwrote {PRED/'synth10k_within_cv_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper metrics: Exact, OBO, MAE, MAE_n on RepProbe-Real.

Parse failure (response, no integer) is scored as 0 then clipped to [0, 40].
Missing (not run / crash / empty error) is excluded, not scored as 0.

  python scripts/rescore_obo.py --fail-as-zero --pool both
  python scripts/rescore_obo.py --fail-as-zero --pool 32k --json data/repcount_nrep/preds/obo_fail_as_zero.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repprobe.paths import REPO_ROOT, REPCOUNT

ROOT = REPO_ROOT
PREDS = REPCOUNT / "preds"
OVR = ROOT / "data/ovr/preds"
BANDS = {"few": (1, 6), "mid": (7, 13), "many": (14, 20)}
# True counts stop at 20, but chat VLMs genuinely answer things like 746 or
# 31500. Those are real outputs, not parse errors, so mean-based metrics are
# reported both raw and with predictions clipped into a plausible range.
CLIP_MAX = 40

# label -> (group, [files]); multiple files are merged shards.
MODELS_10K: dict[str, tuple[str, list[str]]] = {
    "Qwen3-VL-8B": ("Chat VLM", ["eval10000__Qwen_Qwen3-VL-8B-Instruct.jsonl"]),
    "InternVL3.5-8B": ("Chat VLM", ["eval10000__OpenGVLab_InternVL3_5-8B-HF.jsonl"]),
    "Cosmos-Reason2-8B": ("Chat VLM", ["eval10000__nvidia_Cosmos-Reason2-8B.jsonl"]),
    "VideoChat-R1.5-7B": ("Chat VLM", ["eval10000__OpenGVLab_VideoChat-R1_5-7B.jsonl"]),
    "Video-R1-7B": ("Chat VLM", ["eval10000__Video-R1_Video-R1-7B.jsonl"]),
    "LLaVA-OneVision-7B": ("Chat VLM", ["eval10000__llava-hf_llava-onevision-qwen2-7b-ov-hf.jsonl"]),
    "LongVILA-R1-7B": (
        "Chat VLM",
        [
            "eval10000__Efficient-Large-Model_LongVILA-R1-7B__shard0.jsonl",
            "eval10000__Efficient-Large-Model_LongVILA-R1-7B__shard1.jsonl",
        ],
    ),
    "Gemini 3.5 Flash (1 FPS)": ("Chat VLM (API)", ["eval10000__gemini-3.5-flash.jsonl"]),
    "Gemini 3.1 Pro (1 FPS)": ("Chat VLM (API)", ["eval10000__gemini-3.1-pro-preview.jsonl"]),
    # produced by score_gemini_fps.py, which folds in every .retryN shard
    "Gemini 3.5 Flash (fps=4/8)": (
        "Chat VLM (API)",
        ["eval10000__gemini-3.5-flash-fps4.merged.jsonl"],
    ),
    "RepNet": ("Specialist", ["eval10000__google_RepNet.jsonl"]),
    "ESCounts (RepCount)": ("Specialist", ["eval10000__EveryShotCounts_ESCounts-RepCount.jsonl"]),
    "Cosmos internal (L35)": ("Frozen probe", ["eval10000__cosmos_hidden_probe.jsonl"]),
}

MODELS_32K: dict[str, tuple[str, list[str]]] = {
    "Qwen3-VL-8B": ("Chat VLM", ["eval32000__Qwen_Qwen3-VL-8B-Instruct.jsonl"]),
    "InternVL3.5-8B": ("Chat VLM", ["eval32000__OpenGVLab_InternVL3_5-8B-HF.jsonl"]),
    "Cosmos-Reason2-8B": ("Chat VLM", ["eval32000__nvidia_Cosmos-Reason2-8B.jsonl"]),
    "VideoChat-R1.5-7B": ("Chat VLM", ["eval32000__OpenGVLab_VideoChat-R1_5-7B.jsonl"]),
    "Video-R1-7B": ("Chat VLM", ["eval32000__Video-R1_Video-R1-7B.jsonl"]),
    "LLaVA-OneVision-7B": ("Chat VLM", ["eval32000__llava-hf_llava-onevision-qwen2-7b-ov-hf.jsonl"]),
    "LongVILA-R1-7B": (
        "Chat VLM",
        [
            "eval32000__Efficient-Large-Model_LongVILA-R1-7B__shard0.jsonl",
            "eval32000__Efficient-Large-Model_LongVILA-R1-7B__shard1.jsonl",
        ],
    ),
    "RepNet": ("Specialist", ["eval32000__google_RepNet.jsonl"]),
    "ESCounts (RepCount)": ("Specialist", ["eval32000__EveryShotCounts_ESCounts-RepCount.jsonl"]),
    "Cosmos internal (L35)": ("Frozen probe", ["eval32000__cosmos_hidden_probe.jsonl"]),
}

EVAL_10K = REPCOUNT / "eval_10000.jsonl"
EVAL_32K = REPCOUNT / "all.jsonl"

MODELS_OVR: dict[str, tuple[str, list[str]]] = {
    "Qwen3-VL-8B": ("Chat VLM", ["qwen_ovr_100.jsonl"]),
    "InternVL3.5-8B": ("Chat VLM", ["ovr100__OpenGVLab_InternVL3_5-8B-HF.jsonl"]),
    "Cosmos-Reason2-8B": ("Chat VLM", ["ovr100__nvidia_Cosmos-Reason2-8B.jsonl"]),
    "VideoChat-R1.5-7B": ("Chat VLM", ["ovr100__OpenGVLab_VideoChat-R1_5-7B.jsonl"]),
    "Video-R1-7B": ("Chat VLM", ["ovr100__Video-R1_Video-R1-7B.jsonl"]),
    "LLaVA-OneVision-7B": ("Chat VLM", ["ovr100__llava-hf_llava-onevision-qwen2-7b-ov-hf.jsonl"]),
    "LongVILA-R1-7B": ("Chat VLM", ["ovr100__Efficient-Large-Model_LongVILA-R1-7B.jsonl"]),
}


def load_merged(base: Path, files: list[str]) -> list[dict] | None:
    rows: dict[str, dict] = {}
    if not (base / files[0]).is_file():
        return None
    for f in files:
        p = base / f
        if not p.is_file():
            continue
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid is None:
                continue
            # later duplicates (retries) win only if they carry a prediction
            if vid not in rows or (rows[vid].get("pred") is None and r.get("pred") is not None):
                rows[vid] = r
    return list(rows.values())


def load_eval_gts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            gt = r.get("count")
            if vid is None or gt is None:
                continue
            rows.append({"video_id": vid, "gt": int(gt)})
    return rows


SKIP_PAPER = {
    "Gemini 3.5 Flash (1 FPS)",
    "Gemini 3.1 Pro (1 FPS)",
    "Gemini 3.5 Flash (fps=4/8)",
}


def classify_src(src: dict | None) -> str:
    """ok | parse_fail | missing. Crashes / not-run are missing, not failures."""
    if src is None:
        return "missing"
    if src.get("pred") is not None:
        return "ok"
    raw = str(src.get("raw_text") or src.get("response") or src.get("text") or "").strip()
    err = src.get("error")
    if err and not raw:
        return "missing"
    if raw:
        return "parse_fail"
    return "missing" if err else "parse_fail"


def align_to_eval(pred_rows: list[dict], eval_rows: list[dict], fail_as: int | None) -> list[dict]:
    """One row per eval clip. Parse failures can be filled; missing stays missing."""
    by_id = {r["video_id"]: r for r in pred_rows if r.get("video_id")}
    out = []
    for rec in eval_rows:
        vid = rec["video_id"]
        gt = int(rec["gt"])
        src = by_id.get(vid)
        kind = classify_src(src)
        pred = None if src is None else src.get("pred")
        if kind == "parse_fail" and fail_as is not None:
            pred = fail_as
        out.append({
            "video_id": vid,
            "gt": gt,
            "pred": pred,
            "kind": kind,
            "fail": kind == "parse_fail",
        })
    return out


def score(rows: list[dict], bands: bool = True, fail_as: int | None = None, clip: bool = False) -> dict:
    total = len(rows)
    filled = []
    n_fail = n_missing = 0
    for r in rows:
        if r.get("gt") is None:
            continue
        kind = r.get("kind")
        pred = r.get("pred")
        if kind is None:
            if pred is None:
                kind = "parse_fail" if fail_as is not None else "missing"
            else:
                kind = "ok"
        if kind == "missing" or (pred is None and kind != "parse_fail"):
            n_missing += 1
            continue
        if kind == "parse_fail" or pred is None:
            if fail_as is None:
                n_fail += 1
                continue
            pred = fail_as
            n_fail += 1
        pred = int(pred)
        if clip:
            pred = min(max(pred, 0), CLIP_MAX)
        filled.append({"gt": int(r["gt"]), "pred": pred})
    ok = filled
    n = len(ok)
    fail = n_fail
    if fail_as is None and not any(r.get("kind") for r in rows):
        ok = [r for r in rows if r.get("pred") is not None and r.get("gt") is not None]
        n = len(ok)
        fail = total - n
        n_missing = fail
        if clip:
            ok = [{"gt": int(r["gt"]), "pred": min(max(int(r["pred"]), 0), CLIP_MAX)} for r in ok]
    if not n:
        return {}
    hits = sum(abs(r["pred"] - r["gt"]) <= 1 for r in ok)
    rel = sorted(abs(r["pred"] - r["gt"]) / max(int(r["gt"]), 1) for r in ok)
    out = {
        "n_total": total,
        "n_parsed": n - fail if fail_as is not None else n,
        "n_scored": n,
        "n_missing": n_missing,
        "fail": fail,
        "fail_as": fail_as,
        "clip": clip,
        "obo": hits / n,
        "obo_all": hits / max(n + n_missing, 1),
        "mae_norm": sum(rel) / n,
        "mae_norm_clip": sum(rel) / n,
        "mae_norm_median": rel[n // 2],
        "mae_raw": sum(abs(r["pred"] - r["gt"]) for r in ok) / n,
        "mae_raw_clip": sum(abs(r["pred"] - r["gt"]) for r in ok) / n,
        "n_over_clip": sum(r["pred"] >= CLIP_MAX for r in ok),
        "max_pred": max(r["pred"] for r in ok),
        "exact": sum(r["pred"] == r["gt"] for r in ok) / n,
    }
    if bands:
        for name, (lo, hi) in BANDS.items():
            v = [r for r in ok if lo <= int(r["gt"]) <= hi]
            out[f"obo_{name}"] = (
                sum(abs(r["pred"] - r["gt"]) <= 1 for r in v) / len(v) if v else None
            )
            out[f"mae_raw_{name}"] = (
                sum(abs(r["pred"] - r["gt"]) for r in v) / len(v) if v else None
            )
            out[f"mae_norm_{name}"] = (
                sum(abs(r["pred"] - r["gt"]) / max(int(r["gt"]), 1) for r in v) / len(v)
                if v else None
            )
            out[f"n_{name}"] = len(v)
    return out


def report(title: str, base: Path, models: dict, bands: bool, eval_rows=None, fail_as=None, clip=False, skip=()) -> dict:
    print(f"\n=== {title} ===")
    hdr = (
        f"{'model':<28} {'group':<15} {'n':>6} {'miss':>5} {'fail':>5} "
        f"{'MAE':>7} {'MAE_n':>7} {'Ex.':>6} {'OBO':>7}"
    )
    if bands:
        hdr += f" {'OBO few':>8} {'OBO mid':>8} {'OBO many':>9}"
    print(hdr)
    out = {}
    rowset = []
    for label, (group, files) in models.items():
        if label in skip:
            continue
        rows = load_merged(base, files)
        if rows is None:
            print(f"{label:<28} {group:<15} {'-- not on disk --'}")
            continue
        if eval_rows is not None:
            rows = align_to_eval(rows, eval_rows, fail_as=fail_as)
        s = score(rows, bands, fail_as=fail_as, clip=clip)
        if not s:
            continue
        s["group"] = group
        out[label] = s
        rowset.append((label, group, s))
    for label, group, s in sorted(rowset, key=lambda x: x[2]["mae_raw"]):
        nshow = s.get("n_scored", s["n_parsed"])
        line = (
            f"{label:<28} {group:<15} {nshow:6d} {s.get('n_missing', 0):5d} {s['fail']:5d} "
            f"{s['mae_raw']:7.3f} {s['mae_norm']:7.3f} {s['exact']:5.1%} {s['obo']:6.1%}"
        )
        if bands:
            line += (
                f" {s['obo_few']:7.1%} {s['obo_mid']:7.1%} {s['obo_many']:8.1%}"
            )
        print(line)
    return out


WEB_KEYS = (
    "obo", "obo_all", "mae_norm", "mae_norm_clip", "mae_norm_median",
    "mae_raw", "mae_raw_clip", "n_over_clip", "max_pred",
    "obo_few", "obo_mid", "obo_many",
)
# rescore label -> existing key in web/scores_10k_all.json
WEB_ALIAS = {"ESCounts (RepCount)": "ESCounts (RepCount)"}


def _fmt_pair(name: str, s: dict) -> None:
    print(
        f"{name:<28} n={s['n_scored']:5d} miss={s.get('n_missing', 0):4d} "
        f"fail={s['fail']:4d} MAE={s['mae_raw']:.3f} MAE_n={s['mae_norm']:.3f} "
        f"Ex={s['exact']:.1%} OBO={s['obo']:.1%}"
    )


def score_fairness_pair(fail_as=0, clip=True) -> dict:
    """Table fairness-both: each answer readout on clips where that answer and the probe exist."""
    eval_rows = load_eval_gts(EVAL_10K)
    probe = load_merged(PREDS, MODELS_10K["Cosmos internal (L35)"][1])
    gen = load_merged(PREDS, MODELS_10K["Cosmos-Reason2-8B"][1])
    qwen = load_merged(PREDS, MODELS_10K["Qwen3-VL-8B"][1])
    qwen_probe = load_merged(PREDS, ["eval10000__qwen_hidden_probe.jsonl"])
    cosmos_probe_ok = {r["video_id"] for r in probe if r.get("pred") is not None}
    qwen_probe_ok = {r["video_id"] for r in qwen_probe if r.get("pred") is not None}

    def present_ids(rows):
        by = {r["video_id"]: r for r in rows if r.get("video_id")}
        out = set()
        for rec in eval_rows:
            kind = classify_src(by.get(rec["video_id"]))
            if kind != "missing":
                out.add(rec["video_id"])
        return out

    cosmos_ids = cosmos_probe_ok & present_ids(gen)
    qwen_ids = qwen_probe_ok & present_ids(qwen)
    cosmos_pair = [r for r in eval_rows if r["video_id"] in cosmos_ids]
    qwen_pair = [r for r in eval_rows if r["video_id"] in qwen_ids]
    out = {}

    print(f"\n=== fairness-both Cosmos pair n={len(cosmos_pair)} (Cosmos L35 probe ∩ Cosmos answer) ===")
    for name, rows, fa, do_clip in (
        ("Cosmos probe L35", probe, None, True),
        ("Cosmos-Reason2-8B gen", gen, fail_as, clip),
        ("Cosmos-Reason2-8B gen unclip", gen, fail_as, False),
    ):
        aligned = align_to_eval(rows, cosmos_pair, fail_as=fail_as)
        if fa is None:
            aligned = [r for r in aligned if r["kind"] != "parse_fail"]
        s = score(aligned, True, fail_as=fa, clip=do_clip)
        s["group"] = "paired"
        out[name] = s
        _fmt_pair(name, s)

    print(f"\n=== fairness-both Qwen pair n={len(qwen_pair)} (Qwen L35 probe ∩ Qwen answer) ===")
    for name, rows, fa, do_clip in (
        ("Qwen probe L35", qwen_probe, None, True),
        ("Qwen3-VL-8B gen", qwen, fail_as, clip),
        ("Qwen3-VL-8B gen unclip", qwen, fail_as, False),
    ):
        aligned = align_to_eval(rows, qwen_pair, fail_as=fail_as)
        if fa is None:
            aligned = [r for r in aligned if r["kind"] != "parse_fail"]
        s = score(aligned, True, fail_as=fa, clip=do_clip)
        s["group"] = "paired"
        out[name] = s
        _fmt_pair(name, s)
    return out


def write_web(scores: dict, path: Path) -> None:
    board = json.loads(path.read_text()) if path.is_file() else {}
    added, updated = [], []
    for label, s in scores.items():
        key = WEB_ALIAS.get(label, label)
        row = board.get(key)
        if row is None:
            added.append(key)
            row = {"group": s["group"]}
        else:
            updated.append(key)
        # coverage and the legacy metric names drift whenever a run is extended
        # or retried, so refresh them rather than only appending the new keys
        row.update({"n": s["n_parsed"], "fail": s["fail"], "exact": s["exact"],
                    "mae": s["mae_raw"], "off1": s["obo"]})
        for k in WEB_KEYS:
            if s.get(k) is not None:
                row[k] = s[k]
        board[key] = row
    path.write_text(json.dumps(board, indent=2) + "\n")
    print(f"\nwrote {path}\n  updated {len(updated)} rows, added {len(added)}: {added}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovr", action="store_true", help="also score the OVR-100 in-the-wild set")
    ap.add_argument("--json", default="", help="write all scores to this path")
    ap.add_argument(
        "--write-web",
        action="store_true",
        help="fold obo/mae_norm/per-band OBO into a JSON board (default: unused)",
    )
    ap.add_argument(
        "--fail-as-zero",
        action="store_true",
        help="score unparseable / missing preds as 0 instead of dropping them",
    )
    ap.add_argument(
        "--pool",
        default="10k",
        choices=("10k", "32k", "both"),
        help="which eval pool to score",
    )
    args = ap.parse_args()
    fail_as = 0 if args.fail_as_zero else None
    clip = bool(args.fail_as_zero)
    skip = SKIP_PAPER if args.fail_as_zero else ()
    tag = "parse-fail as 0, then clip [0,40]; missing excluded" if args.fail_as_zero else "exclude failures"

    out = {}
    if args.pool in ("10k", "both"):
        eval_10k = load_eval_gts(EVAL_10K) if args.fail_as_zero else None
        title = f"RepCount n-rep, 10k balanced ({tag})"
        out["repcount_10k"] = report(
            title, PREDS, MODELS_10K, True, eval_10k, fail_as, clip=clip, skip=skip
        )
    if args.pool in ("32k", "both"):
        eval_32k = load_eval_gts(EVAL_32K) if args.fail_as_zero else None
        title = f"RepCount n-rep, full pool 32k ({tag})"
        out["repcount_32k"] = report(
            title, PREDS, MODELS_32K, True, eval_32k, fail_as, clip=clip, skip=skip
        )
    if args.fail_as_zero and args.pool in ("10k", "both"):
        out["fairness_both"] = score_fairness_pair(fail_as=0, clip=True)
    if args.ovr:
        out["ovr_100"] = report("OVR-100 in-the-wild (k=2-9)", OVR, MODELS_OVR, False)
    if args.write_web and "repcount_10k" in out:
        write_web(out["repcount_10k"], ROOT / "data/repcount_nrep/preds/scores_board.json")

    print(
        "\nParse failure = response exists but parse_count is None → pred=0, then clip [0,40]."
        "\nMissing = not run / crash / empty+error → excluded, not scored as 0."
        "\nGemini omitted under --fail-as-zero."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

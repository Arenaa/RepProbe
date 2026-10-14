#!/usr/bin/env python3
"""Linear count probe on Cosmos-Reason2-8B / Qwen3-VL frozen hidden states.

Hooks language-model residual blocks at the last prompt token (the state the
LM head reads before generate). By default probes three depths:

  early  = layers.0
  mid    = layers[N//2]   (L18 for 36-layer Qwen3-VL)
  last   = layers[N-1]    (L35 — same site as the main Cosmos readout claim)

If the last-layer probe recovers k while generation does not — especially at
high k — the count is encoded internally and lost at generative readout.
Comparing early/mid/last shows *where* the count signal appears in the LM.

Train protocol: 50 leftover train clips per k.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def _trace_dict():
    try:
        from baukit.nethook import TraceDict
        return TraceDict
    except ImportError as exc:
        path = os.environ.get("BAUKIT_NETHOOK")
        if not path:
            raise ImportError(
                "Install baukit (`pip install baukit`) or set BAUKIT_NETHOOK to nethook.py"
            ) from exc
        import importlib.util

        spec = importlib.util.spec_from_file_location("baukit_nethook", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TraceDict


MODEL_ID = "nvidia/Cosmos-Reason2-8B"
QUESTION = "Count the number of repetitions in this video. Return only an integer."
FPS = 4.0
VIDEO_FPS: float | None = 4.0
VIDEO_NFRAMES: int | None = None
LAYER_RE = re.compile(r"language_model\.layers\.(\d+)$")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def stratified(rows: list[dict], per_k: int, seed: int) -> list[dict]:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for rec in rows:
        by_k[int(rec["count"])].append(rec)
    rng = __import__("random").Random(seed)
    out = []
    positions = ["start", "middle", "end"]
    for k in range(1, 21):
        recs = list(by_k[k])
        if len(recs) < per_k:
            raise SystemExit(f"count={k} has {len(recs)} clips, need {per_k}")
        rng.shuffle(recs)
        picked, used = [], set()
        for pos in positions * per_k:
            for r in recs:
                if r["video_id"] in used:
                    continue
                if r.get("position") == pos:
                    picked.append(r)
                    used.add(r["video_id"])
                    break
            if len(picked) >= per_k:
                break
        for r in recs:
            if len(picked) >= per_k:
                break
            if r["video_id"] not in used:
                picked.append(r)
                used.add(r["video_id"])
        out.extend(picked[:per_k])
    return out


def resolve_video(data_root: Path, rec: dict) -> str:
    rel = rec.get("video") or ""
    return rel if os.path.isabs(rel) else str(data_root / rel)


def to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def lm_layer_names(model) -> list[str]:
    names = []
    for n, _ in model.named_modules():
        m = LAYER_RE.search(n)
        if m:
            names.append((int(m.group(1)), n))
    names.sort()
    if not names:
        raise SystemExit("no language_model.layers.* modules found")
    return [n for _, n in names]


def pick_lm_layers(layers_all: list[str], spec: str) -> list[str]:
    """Select language residual blocks.

    ``spec`` is a comma list of roles and/or indices, e.g.
      early,mid,last   -> first, middle, final decoder block (default)
      0,17,35          -> explicit indices into language_model.layers.*
    """
    n = len(layers_all)
    if n == 0:
        raise SystemExit("empty layer list")
    role_ix = {
        "early": 0,
        "first": 0,
        "mid": n // 2,
        "middle": n // 2,
        "last": n - 1,
        "final": n - 1,
    }
    chosen: list[int] = []
    for tok in (t.strip().lower() for t in spec.split(",") if t.strip()):
        if tok in role_ix:
            chosen.append(role_ix[tok])
        else:
            try:
                ix = int(tok)
            except ValueError as e:
                raise SystemExit(f"bad --layers token {tok!r}; use early/mid/last or int") from e
            if not (0 <= ix < n):
                raise SystemExit(f"layer index {ix} out of range [0,{n-1}]")
            chosen.append(ix)
    # stable unique, preserve order of first occurrence
    seen: set[int] = set()
    out = []
    for ix in chosen:
        if ix not in seen:
            seen.add(ix)
            out.append(layers_all[ix])
    return out


def layer_role(name: str, layers_all: list[str]) -> str:
    try:
        ix = layers_all.index(name)
    except ValueError:
        m = re.search(r"layers\.(\d+)", name)
        return f"L{m.group(1)}" if m else name
    n = len(layers_all)
    if ix == 0:
        return "early"
    if ix == n - 1:
        return "last"
    if ix == n // 2:
        return "mid"
    return f"L{ix}"


def as_hidden(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    return output


FRAME_FACTOR = 2  # qwen_vl_utils rounds nframes to this factor


def video_frame_count(video_path: str) -> int:
    try:
        from decord import VideoReader, cpu

        return len(VideoReader(video_path, ctx=cpu(0)))
    except Exception:
        import cv2

        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return max(n, 1)


def cap_video_nframes(requested: int, video_path: str) -> int:
    """Cap nframes so smart_nframes' round-by-2 stays within the video."""
    n_avail = video_frame_count(video_path)
    capped = min(int(requested), n_avail)
    capped = (capped // FRAME_FACTOR) * FRAME_FACTOR  # floor, not round
    return max(FRAME_FACTOR, capped)


def prepare_inputs(processor, video_path: str, question: str):
    from qwen_vl_utils import process_vision_info

    video_ele: dict = {"type": "video", "video": video_path}
    if VIDEO_NFRAMES is not None:
        video_ele["nframes"] = cap_video_nframes(int(VIDEO_NFRAMES), video_path)
    elif VIDEO_FPS is not None:
        video_ele["fps"] = float(VIDEO_FPS)
    else:
        video_ele["fps"] = float(FPS)
    messages = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, metadatas = zip(*videos)
            videos, metadatas = list(videos), list(metadatas)
        else:
            metadatas = None
        return processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=metadatas,
            return_tensors="pt",
            do_resize=False,
            **(video_kwargs or {}),
        )
    except TypeError:
        images, videos = process_vision_info(messages)
        return processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )


def pool_trace(tr, name: str) -> tuple[np.ndarray, np.ndarray]:
    h = as_hidden(tr[name].output)
    # [B, T, D]
    last = h[0, -1].float().cpu().numpy().astype(np.float32)
    mean = h[0].float().mean(0).cpu().numpy().astype(np.float32)
    return last, mean


def encode_one(model, processor, layers: list[str], video_path: str, question: str) -> dict[str, np.ndarray]:
    TraceDict = _trace_dict()

    device = next(model.parameters()).device
    inputs = to_device(prepare_inputs(processor, video_path, question), device)
    with torch.inference_mode():
        with TraceDict(model, layers, retain_output=True, detach=True, clone=False) as tr:
            model(**inputs, use_cache=False)
        out = {}
        for name in layers:
            last, mean = pool_trace(tr, name)
            out[f"{name}::last"] = last
            out[f"{name}::mean"] = mean
    return out


def load_cache(paths: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Decompress each array once. Indexing a compressed npz per-row re-inflates it."""
    cache: dict[str, dict[str, np.ndarray]] = {}
    for p in paths:
        if not p.is_file() or p.stat().st_size < 1_000_000:
            continue
        try:
            blob = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"skip cache {p}: {type(e).__name__}: {e}", flush=True)
            continue
        keys = [k for k in blob.files if k != "ids"]
        ids = [str(v) for v in blob["ids"]]
        arrs = {k: np.asarray(blob[k]) for k in keys}
        for i, vid in enumerate(ids):
            cache[vid] = {k: arrs[k][i] for k in keys}
        print(f"cache +{len(ids)} from {p} total={len(cache)}", flush=True)
    return cache


def save_cache(path: Path, cache: dict[str, dict[str, np.ndarray]]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()))
    keys = sorted(next(iter(cache.values())).keys())
    kw = {k: np.stack([cache[i][k] for i in ids]) for k in keys}
    np.savez_compressed(path, ids=ids, **kw)
    print(f"saved cache {path} n={len(ids)} keys={keys}", flush=True)


def encode_rows(model, processor, layers, rows, data_root, cache, cache_path, tag: str) -> None:
    t0 = time.time()
    n_new = 0
    for i, rec in enumerate(rows, 1):
        vid = rec["video_id"]
        if vid in cache:
            if i == 1 or i % 50 == 0 or i == len(rows):
                print(f"{tag} {i}/{len(rows)} cached  {time.time()-t0:.0f}s", flush=True)
            continue
        root = Path(rec["_data_root"]) if rec.get("_data_root") else data_root
        path = resolve_video(root, rec)
        try:
            cache[vid] = encode_one(
                model, processor, layers, path, rec.get("question") or QUESTION
            )
            n_new += 1
        except Exception as e:
            print(f"encode fail {vid}: {type(e).__name__}: {e}", flush=True)
            continue
        if n_new > 0 and n_new % 100 == 0:
            save_cache(cache_path, cache)
        if i == 1 or i % 20 == 0 or i == len(rows):
            print(f"{tag} {i}/{len(rows)} new={n_new}  {time.time()-t0:.0f}s", flush=True)
    save_cache(cache_path, cache)


def stack_feat(rows, cache, feat_key: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs, ys, kept = [], [], []
    for rec in rows:
        vid = rec["video_id"]
        if vid not in cache or feat_key not in cache[vid]:
            continue
        xs.append(cache[vid][feat_key])
        ys.append(int(rec["count"]))
        kept.append(rec)
    return np.stack(xs), np.asarray(ys, dtype=np.int64), kept


def per_k(y_true, y_pred) -> dict:
    out = {}
    for k in range(1, 21):
        m = y_true == k
        n = int(m.sum())
        if n == 0:
            out[str(k)] = dict(n=0, acc=None, mae=None)
            continue
        out[str(k)] = dict(
            n=n,
            acc=float(np.mean(y_pred[m] == y_true[m])),
            mae=float(np.mean(np.abs(y_pred[m] - y_true[m]))),
        )
    return out


def fit_one(
    x_tr,
    y_tr,
    x_ev,
    y_ev,
    kept,
    feat_key: str,
    out_path: Path,
    role: str = "",
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.fit_transform(x_tr), y_tr)
    pred = clf.predict(scaler.transform(x_ev))
    mae = float(np.mean(np.abs(pred - y_ev)))
    exact = float(np.mean(pred == y_ev))
    ob1 = float(np.mean(np.abs(pred - y_ev) <= 1))
    tag = f"{role} " if role else ""
    print(
        f"{tag}{feat_key}  MAE={mae:.3f} exact={exact:.3f} offby1={ob1:.3f} n={len(pred)}",
        flush=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec, p, g in zip(kept, pred.tolist(), y_ev.tolist()):
            f.write(
                json.dumps(
                    {
                        "video_id": rec.get("video_id"),
                        "model": f"{MODEL_ID}+hidden_probe",
                        "pred": int(p),
                        "gt": int(g),
                        "k": int(g),
                        "feat": feat_key,
                        "layer_role": role or None,
                        "position": rec.get("position"),
                        "action_type": rec.get("action_type"),
                        "raw_text": "hidden_linear_probe",
                        "error": None,
                        "sec": None,
                    }
                )
                + "\n"
            )
    bands = {}
    for name, (lo, hi) in (("few", (1, 6)), ("mid", (7, 13)), ("many", (14, 20))):
        m = (y_ev >= lo) & (y_ev <= hi)
        if not m.any():
            bands[name] = None
            continue
        bands[name] = dict(
            n=int(m.sum()),
            mae=float(np.mean(np.abs(pred[m] - y_ev[m]))),
            exact=float(np.mean(pred[m] == y_ev[m])),
            offby1=float(np.mean(np.abs(pred[m] - y_ev[m]) <= 1)),
        )
    return dict(
        mae=mae,
        exact=exact,
        offby1=ob1,
        n=int(len(pred)),
        by_k=per_k(y_ev, pred),
        bands=bands,
        feat=feat_key,
        role=role or None,
    )


def main() -> None:
    global MODEL_ID, VIDEO_FPS, VIDEO_NFRAMES
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/repcount_nrep")
    ap.add_argument("--train-jsonl", default="data/repcount_nrep/train.jsonl")
    ap.add_argument("--eval-jsonl", default="data/repcount_nrep/eval_10000.jsonl")
    ap.add_argument("--per-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train+eval rows")
    ap.add_argument("--mode", choices=("encode", "fit", "all"), default="all")
    ap.add_argument("--shard", default="")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--fps", type=float, default=None, help="video fps (default 4 for Cosmos)")
    ap.add_argument("--nframes", type=int, default=None, help="fixed frame budget; overrides --fps")
    ap.add_argument(
        "--layers",
        default="early,mid,last",
        help="language residual blocks to hook: early,mid,last and/or integer indices",
    )
    ap.add_argument("--out", default="data/repcount_nrep/preds/eval10000__cosmos_hidden_probe.jsonl")
    ap.add_argument("--metrics", default="data/repcount_nrep/preds/cosmos_hidden_probe_metrics.json")
    ap.add_argument("--emb-cache", default="data/repcount_nrep/preds/eval10000__cosmos_hidden_probe.emb.npz")
    ap.add_argument(
        "--eval-data-root",
        default="",
        help="if set, resolve eval videos under this root (train stays --data-root)",
    )
    args = ap.parse_args()

    MODEL_ID = args.model_id
    if args.nframes is not None:
        VIDEO_NFRAMES = int(args.nframes)
        VIDEO_FPS = None
    elif args.fps is not None:
        VIDEO_FPS = float(args.fps)
        VIDEO_NFRAMES = None

    data_root = Path(args.data_root)
    eval_root = Path(args.eval_data_root) if args.eval_data_root else data_root
    eval_rows = load_jsonl(Path(args.eval_jsonl))
    eval_ids = {r["video_id"] for r in eval_rows}
    # Stratified leftover outside eval when possible; if eval is the
    # full corpus (train ⊂ eval), keep the stratified picks and warn (mild leakage
    # on those ~per-k×20 clips only).
    train_all = load_jsonl(Path(args.train_jsonl))
    train_rows = stratified(train_all, args.per_k, args.seed)
    filtered = [r for r in train_rows if r["video_id"] not in eval_ids]
    if filtered:
        train_rows = filtered
    elif eval_ids:
        print(
            f"warn: eval jsonl covers all stratified train picks; "
            f"keeping {len(train_rows)} train clips (full-corpus eval)",
            flush=True,
        )
    if args.limit:
        train_rows, eval_rows = train_rows[: max(20, args.limit // 5)], eval_rows[: args.limit]
        print(f"limit train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    for r in train_rows:
        r["_data_root"] = str(data_root)
    for r in eval_rows:
        r["_data_root"] = str(eval_root)

    encode_all = train_rows + eval_rows
    si = sn = None
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        encode_all = [r for j, r in enumerate(encode_all) if j % sn == si]
        print(f"shard {args.shard}: {len(encode_all)} rows", flush=True)

    cache_path = Path(args.emb_cache)
    if args.shard:
        cache_path = cache_path.with_name(cache_path.stem + f"__shard{si}" + cache_path.suffix)
    pred_dir = cache_path.parent
    if "__shard" in cache_path.stem:
        shard_glob = cache_path.stem.split("__shard")[0] + "__shard*" + cache_path.suffix
    else:
        shard_glob = cache_path.stem + "__shard*" + cache_path.suffix
    shard_caches = sorted(pred_dir.glob(shard_glob))
    cache: dict = {}
    if args.mode == "encode":
        cache = load_cache([cache_path])
    elif args.mode == "all":
        cache = load_cache(shard_caches + [cache_path])

    layers_all_names: list[str] = []
    if args.mode in ("encode", "all"):
        n_vis = torch.cuda.device_count()
        if n_vis != 1:
            raise SystemExit(f"expected 1 visible GPU, got {n_vis}")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(
            f"loading {MODEL_ID}  fps={VIDEO_FPS} nframes={VIDEO_NFRAMES}",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model.eval()
        layers_all_names = lm_layer_names(model)
        layers = pick_lm_layers(layers_all_names, args.layers)
        roles = {n: layer_role(n, layers_all_names) for n in layers}
        print(
            f"hooking {len(layers)}/{len(layers_all_names)} LM blocks: "
            + ", ".join(f"{roles[n]}={n}" for n in layers),
            flush=True,
        )
        encode_rows(model, processor, layers, encode_all, data_root, cache, cache_path, tag="encode")
        del model
        torch.cuda.empty_cache()

    if args.mode in ("fit", "all") and not args.shard:
        if args.mode == "fit" or not cache:
            globs = shard_caches or sorted(pred_dir.glob(shard_glob))
            cache = load_cache(globs)
        feat_keys = sorted({k for rec in cache.values() for k in rec})
        last_keys = [k for k in feat_keys if k.endswith("::last")]

        def _layer_ix(k: str) -> int:
            m = re.search(r"layers\.(\d+)", k)
            return int(m.group(1)) if m else -1

        last_keys.sort(key=_layer_ix)
        if not last_keys:
            raise SystemExit(f"no last-token features in cache n={len(cache)} keys={feat_keys}")
        primary = last_keys[-1]
        print(f"primary feat {primary}", flush=True)

        if not layers_all_names:
            layers_all_names = sorted(
                {re.sub(r"::(last|mean)$", "", k) for k in feat_keys},
                key=_layer_ix,
            )

        metrics = {
            "primary": primary,
            "model_id": MODEL_ID,
            "layers_spec": args.layers,
            "train_n": len(train_rows),
            "eval_n": len(eval_rows),
            "eval_jsonl": str(args.eval_jsonl),
            "probes": {},
            "by_role": {},
        }
        mean_last = primary.replace("::last", "::mean")
        to_fit = list(last_keys)
        if mean_last in feat_keys:
            to_fit.append(mean_last)
        for key in to_fit:
            x_tr, y_tr, _ = stack_feat(train_rows, cache, key)
            x_ev, y_ev, kept = stack_feat(eval_rows, cache, key)
            if len(x_tr) < 20 or len(x_ev) < 10:
                print(f"skip {key}: train={len(x_tr)} eval={len(x_ev)}", flush=True)
                continue
            base = re.sub(r"::(last|mean)$", "", key)
            role = layer_role(base, layers_all_names)
            if key.endswith("::mean"):
                role = f"{role}_mean"
            out = Path(args.out) if key == primary else Path(args.out).with_name(
                Path(args.out).stem + "__L" + str(_layer_ix(key)) + "_" + key.split("::")[-1] + ".jsonl"
            )
            m = fit_one(x_tr, y_tr, x_ev, y_ev, kept, key, out, role=role)
            metrics["probes"][key] = m
            metrics["by_role"][role] = {
                "feat": key,
                "mae": m["mae"],
                "exact": m["exact"],
                "offby1": m["offby1"],
                "n": m["n"],
                "bands": m["bands"],
            }
        Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"wrote {args.metrics}", flush=True)
        print("\n=== layer sweep (last-token residual) ===", flush=True)
        print(f"{'role':<12} {'MAE':>7} {'exact':>7} {'OBO':>7} {'n':>5}", flush=True)
        for role in ("early", "mid", "last"):
            r = metrics["by_role"].get(role)
            if not r:
                continue
            print(
                f"{role:<12} {r['mae']:7.3f} {100*r['exact']:6.1f}% {100*r['offby1']:6.1f}% {r['n']:5d}",
                flush=True,
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Number-token prior at the Cosmos / Qwen3-VL LM head.

At the last prompt token, take next-token logits, score integer digit
sequences (k=0..KMAX), and write per-clip P(true k), argmax_int, entropy.

k>=10 tokenizes as two digits ('14' -> '1'+'4'), so P(k) is the sum of
log-probs along that sequence (second digit teacher-forced via KV cache).

    CUDA_VISIBLE_DEVICES=0 python3 eval_number_prior.py --records data/repcount_nrep/smoke_200.jsonl --out data/repcount_nrep/preds/smoke200__number_prior.jsonl
    CUDA_VISIBLE_DEVICES=0 python3 eval_number_prior.py --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from repprobe.probe import hidden as cos

MODEL_ID = "nvidia/Cosmos-Reason2-8B"
QUESTION = cos.QUESTION
KMAX = 40
DIGIT_IDS = {d: 15 + d for d in range(10)}  # '0'..'9' on the Qwen tokenizer


def already_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {r["video_id"] for r in cos.load_jsonl(path) if r.get("video_id") and r.get("error") is None}


def digits_of(k: int) -> list[int]:
    return [DIGIT_IDS[int(ch)] for ch in str(int(k))]


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1)


def score_from_first(logp0: torch.Tensor, logp1: dict[int, torch.Tensor], k: int) -> float:
    ids = digits_of(k)
    lp = float(logp0[ids[0]])
    if len(ids) == 1:
        return lp
    second = logp1.get(ids[0])
    if second is None:
        return float("-inf")
    return lp + float(second[ids[1]])


def continue_digit(model, inputs, past, first_id: int, device):
    """One-token continuation. Falls back to a full re-forward if cache fails."""
    next_ids = torch.tensor([[first_id]], device=device)
    attn = inputs.get("attention_mask")
    kwargs = dict(input_ids=next_ids, past_key_values=past, use_cache=True)
    if attn is not None:
        kwargs["attention_mask"] = torch.cat(
            [attn, torch.ones(attn.shape[0], 1, device=device, dtype=attn.dtype)], dim=1
        )
    try:
        out = model(**kwargs)
        return out.logits[0, -1]
    except Exception:
        ext = {k: v for k, v in inputs.items() if torch.is_tensor(v)}
        ext["input_ids"] = torch.cat([inputs["input_ids"], next_ids], dim=1)
        if "attention_mask" in ext:
            ext["attention_mask"] = torch.cat(
                [ext["attention_mask"], torch.ones(1, 1, device=device, dtype=ext["attention_mask"].dtype)],
                dim=1,
            )
        out = model(**ext, use_cache=False)
        return out.logits[0, -1]


def score_clip(model, processor, video_path: str, question: str, true_k: int, device):
    inputs = cos.to_device(cos.prepare_inputs(processor, video_path, question), device)
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        logits0 = out.logits[0, -1]
        logp0 = log_softmax(logits0)
        past = out.past_key_values

        first_needed = sorted({digits_of(k)[0] for k in range(0, KMAX + 1) if k >= 10})
        logp1: dict[int, torch.Tensor] = {}
        for fid in first_needed:
            logits1 = continue_digit(model, inputs, past, fid, device)
            logp1[fid] = log_softmax(logits1)

        scores = torch.tensor(
            [score_from_first(logp0, logp1, k) for k in range(0, KMAX + 1)],
            dtype=torch.float32,
        )
        # softmax over the integer set 0..KMAX
        mass = torch.softmax(scores, dim=0)
        argmax_int = int(torch.argmax(scores).item())
        p_true = float(mass[true_k]) if 0 <= true_k <= KMAX else 0.0
        p_argmax = float(mass[argmax_int])
        entropy = float(-(mass * (mass.clamp_min(1e-12).log())).sum())

        digit_mass = torch.softmax(torch.stack([logp0[DIGIT_IDS[d]] for d in range(10)]), dim=0)
        first_digit_true = int(str(true_k)[0])
        p_first_digit = float(digit_mass[first_digit_true])
        argmax_digit = int(torch.argmax(digit_mass).item())

        unres = int(torch.argmax(logits0).item())
        top5 = torch.topk(mass, 5)
        top5_int = [
            {"k": int(i), "p": float(p)}
            for i, p in zip(top5.indices.tolist(), top5.values.tolist())
        ]

    return dict(
        p_true=p_true,
        argmax_int=argmax_int,
        p_argmax=p_argmax,
        entropy=entropy,
        p_first_digit=p_first_digit,
        argmax_digit=argmax_digit,
        unrestricted_argmax_id=unres,
        top5=top5_int,
        mass={str(k): float(mass[k]) for k in range(0, KMAX + 1)},
    )


def text_prior(model, processor, device) -> dict:
    """Unembedding prior: text-only prompt, same integer-set softmax."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": "The number of repetitions is"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt")
    inputs = cos.to_device(inputs, device)
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        logp0 = log_softmax(out.logits[0, -1])
        past = out.past_key_values
        first_needed = sorted({digits_of(k)[0] for k in range(0, KMAX + 1) if k >= 10})
        logp1 = {}
        for fid in first_needed:
            logits1 = continue_digit(model, inputs, past, fid, device)
            logp1[fid] = log_softmax(logits1)
        scores = torch.tensor(
            [score_from_first(logp0, logp1, k) for k in range(0, KMAX + 1)],
            dtype=torch.float32,
        )
        mass = torch.softmax(scores, dim=0)
    return {str(k): float(mass[k]) for k in range(0, KMAX + 1)}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--data-root", default="data/repcount_nrep")
    ap.add_argument("--records", default="data/repcount_nrep/eval_10000.jsonl")
    ap.add_argument("--out", default="data/repcount_nrep/preds/eval10000__number_prior.jsonl")
    ap.add_argument("--text-prior-out", default="data/repcount_nrep/preds/number_unembed_prior.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--skip-text-prior", action="store_true")
    ap.add_argument("--fps", type=float, default=None, help="video fps (default: keep cos.VIDEO_FPS)")
    ap.add_argument("--nframes", type=int, default=None, help="fixed frame budget; overrides --fps")
    args = ap.parse_args()

    if args.nframes is not None:
        cos.VIDEO_NFRAMES = int(args.nframes)
        cos.VIDEO_FPS = None
    elif args.fps is not None:
        cos.VIDEO_FPS = float(args.fps)
        cos.VIDEO_NFRAMES = None

    n_vis = torch.cuda.device_count()
    if n_vis != 1:
        raise SystemExit(f"expected 1 visible GPU, got {n_vis}")
    device = torch.device("cuda:0")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  model={args.model_id}  "
        f"fps={cos.VIDEO_FPS} nframes={cos.VIDEO_NFRAMES}",
        flush=True,
    )

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    if not args.skip_text_prior:
        prior = text_prior(model, processor, device)
        Path(args.text_prior_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.text_prior_out).write_text(json.dumps({"model": args.model_id, "p": prior}, indent=2) + "\n")
        top = sorted(prior.items(), key=lambda kv: -kv[1])[:8]
        print(f"text prior top: {top}", flush=True)

    recs = cos.load_jsonl(Path(args.records))
    if args.limit:
        recs = recs[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    todo = [r for r in recs if r.get("video_id") not in done]
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        todo = [r for j, r in enumerate(todo) if j % sn == si]
        print(f"shard {args.shard}: {len(todo)} remaining", flush=True)
    print(f"clips {len(recs)} skip {len(done)} todo {len(todo)} -> {out_path}", flush=True)

    data_root = Path(args.data_root)
    t0 = time.time()
    n_ok = n_err = 0
    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            vid = rec["video_id"]
            path = cos.resolve_video(data_root, rec)
            gt = int(rec["count"])
            t1 = time.time()
            try:
                scored = score_clip(
                    model, processor, path, rec.get("question") or QUESTION, gt, device
                )
                err = None
                n_ok += 1
            except Exception as e:
                scored = {}
                err = f"{type(e).__name__}: {e}"
                n_err += 1
                if "OutOfMemory" in type(e).__name__:
                    torch.cuda.empty_cache()
            row = {
                "video_id": vid,
                "model": args.model_id + "+number_prior",
                "gt": gt,
                "k": gt,
                "pred": scored.get("argmax_int"),
                "p_true": scored.get("p_true"),
                "argmax_int": scored.get("argmax_int"),
                "p_argmax": scored.get("p_argmax"),
                "entropy": scored.get("entropy"),
                "p_first_digit": scored.get("p_first_digit"),
                "argmax_digit": scored.get("argmax_digit"),
                "unrestricted_argmax_id": scored.get("unrestricted_argmax_id"),
                "top5": scored.get("top5"),
                "mass": scored.get("mass"),
                "position": rec.get("position"),
                "action_type": rec.get("action_type"),
                "error": err,
                "sec": round(time.time() - t1, 3),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if i == 1 or i % 10 == 0 or i == len(todo):
                print(
                    f"[{i}/{len(todo)}] {vid} k={gt} argmax={row['argmax_int']} "
                    f"p_true={row['p_true']} ok={n_ok} err={n_err} {time.time()-t0:.0f}s",
                    flush=True,
                )
    print(f"done ok={n_ok} err={n_err} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Count-eval for RepCount n-rep clips. One model, one visible GPU.

Writes one JSONL line per clip (resume-safe). Parse the last integer after CoT.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

QUESTION = "Count the number of repetitions in this video. Return only an integer."

REGISTRY = {
    "Qwen/Qwen3-VL-8B-Instruct": dict(nframes=16, max_new_tokens=32),
    "OpenGVLab/InternVL3_5-8B-HF": dict(nframes=16, max_new_tokens=32),
    "OpenGVLab/VideoChat-R1_5-7B": dict(nframes=16, max_new_tokens=512),
    "Video-R1/Video-R1-7B": dict(nframes=16, max_new_tokens=512),
    "Efficient-Large-Model/LongVILA-R1-7B": dict(nframes=16, max_new_tokens=512),
    "nvidia/Cosmos-Reason2-8B": dict(fps=4, nframes=None, max_new_tokens=1024),
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": dict(nframes=16, max_new_tokens=32),
}

INT_RE = re.compile(r"-?\d+")
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_count(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if m:
        inner = m.group(1).strip().strip(".,;").lower()
        if inner in NUMBER_WORDS:
            return NUMBER_WORDS[inner]
        nums = INT_RE.findall(m.group(1))
        if nums:
            return int(nums[-1])
    for line in reversed(text.strip().splitlines()):
        s = line.strip().strip(".,;")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if s.lower() in NUMBER_WORDS:
            return NUMBER_WORDS[s.lower()]
    # Whole-string number-word only (Qwen/InternVL overwrite to "nineteen").
    # Do not hunt for "one" inside Video-R1 prose — that would fabricate a count.
    whole = text.strip().strip(".,;").lower()
    if whole in NUMBER_WORDS:
        return NUMBER_WORDS[whole]
    nums = INT_RE.findall(text)
    return int(nums[-1]) if nums else None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_done(out_path: Path) -> set[str]:
    done = set()
    if not out_path.is_file():
        return done
    for rec in load_jsonl(out_path):
        vid = rec.get("video_id")
        if vid:
            done.add(vid)
    return done


def sample_frames(path: str, nframes: int):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video {path}")
    n = min(nframes, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idxs:
        rgb = cv2.cvtColor(frames[int(i)], cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(rgb))
    return out


def available_frames(path: str) -> int:
    """qwen-vl-utils rejects nframes above the clip's own frame count, and the
    decord->torchvision fallback is broken here, so the request has to be capped."""
    import cv2

    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def capped_nframes(path: str, nframes: int, multiple: int = 2) -> int:
    """cv2 can overcount by one vs decord, so leave a margin, then round down to a
    multiple the model's temporal patching accepts: 2 for Qwen-family, 4 for VILA,
    whose video tower reshapes to [-1, 4, h, w, d] and raises on any other count."""
    have = available_frames(path)
    n = min(int(nframes), have - 1) if have else int(nframes)
    n = max(multiple, n - (n % multiple))
    return n


def to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def decode_new(processor, inputs, generated):
    in_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    trimmed = [out[len(inp) :] for inp, out in zip(in_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def make_smoke(split_path: Path, per_k: int, seed: int, out_path: Path) -> None:
    by_k: dict[int, list[dict]] = defaultdict(list)
    for rec in load_jsonl(split_path):
        by_k[int(rec["count"])].append(rec)
    rng = random.Random(seed)
    selected = []
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
        selected.extend(picked[:per_k])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} clips -> {out_path}")


def load_model(model_id: str, processor_id: str | None = None, device_map: str = "cuda:0"):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    print(f"loading {model_id} (device_map={device_map})", flush=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "") or ""
    arch = " ".join(getattr(cfg, "architectures", None) or [])
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if model_type == "vila" or "VILA" in arch:
        from transformers import AutoModel

        kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(model_id, **kwargs)
        if device_map != "auto":
            # resize_token_embeddings can leave the new table on CPU
            model.to(device_map)
        model.eval()
        print(f"ready model_type=vila device={device_map}", flush=True)
        return model, None, "vila"
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception as e:
        print(f"AutoModelForImageTextToText failed ({e}); trying AutoModel", flush=True)
        from transformers import AutoModel

        kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(model_id, **kwargs)
    proc_id = processor_id or model_id
    processor = AutoProcessor.from_pretrained(proc_id, trust_remote_code=True)
    model.eval()
    print(
        f"ready model_type={model_type} processor={proc_id} device={model.device}",
        flush=True,
    )
    return model, processor, model_type


def build_generate_inputs(
    model,
    processor,
    model_type: str,
    video_path: str,
    question: str,
    nframes: int | None,
    fps: float | None,
):
    """Tokenize one clip for ``model.generate``. VILA has no tensor path."""
    nframes = nframes or 16
    if model_type == "vila":
        raise RuntimeError("vila has no tensor generate inputs")
    # InternVL's video processor defaults to 384²; the vision tower + pixel-shuffle
    # path is built for the 448² image grid. 384 yields a 27×27 map that crashes.
    if model_type == "internvl":
        vp = getattr(processor, "video_processor", None)
        ip = getattr(processor, "image_processor", None)
        if vp is not None and ip is not None and getattr(ip, "size", None):
            vp.size = ip.size
    video_ele: dict = {"type": "video", "video": video_path}
    if fps is not None:
        video_ele["fps"] = float(fps)
    else:
        video_ele["nframes"] = capped_nframes(video_path, nframes)
    messages = [{
        "role": "user",
        "content": [video_ele, {"type": "text", "text": question}],
    }]

    if "qwen" in model_type:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        patch = 16 if "qwen3" in model_type else 14
        try:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=patch,
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

    extra = {}
    if fps is not None:
        extra["fps"] = fps
    else:
        extra["num_frames"] = nframes
        extra["fps"] = None
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pass
    except Exception:
        pass

    if "qwen2" in model_type or "qwen2_5" in model_type or "qwen2_vl" in model_type:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        return processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )

    frames = sample_frames(video_path, nframes)
    conversation = [{
        "role": "user",
        "content": [{"type": "video"}] + [{"type": "image"}] * 0 + [
            {"type": "text", "text": question}
        ],
    }]
    try:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        try:
            return processor(text=prompt, videos=[frames], padding=True, return_tensors="pt")
        except TypeError:
            content = [{"type": "image", "image": im} for im in frames]
            content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return processor(text=[prompt], images=frames, padding=True, return_tensors="pt")
    except Exception as e:
        content = [{"type": "image", "image": im} for im in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            return processor(text=[text], images=frames, padding=True, return_tensors="pt")
        except Exception as e2:
            raise RuntimeError(f"all inference paths failed: {e} | {e2}") from e2


def infer_one(model, processor, model_type: str, video_path: str, question: str,
              nframes: int | None, fps: float | None, max_new_tokens: int) -> str:
    nframes = nframes or 16
    if model_type == "vila":
        if fps is not None:
            model.config.fps = float(fps)
        else:
            model.config.num_video_frames = capped_nframes(video_path, nframes, 4)
            model.config.fps = 0
        prompt = (
            "You are a helpful assistant. The user asks a question, and then you solves it.\n\n"
            "Please first think deeply about the question based on the given video, and then provide "
            "the final answer. The reasoning process and answer are enclosed within <think> </think> "
            "and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
            f"<answer> answer here </answer>.\n\n Question: {question}"
        )
        with torch.inference_mode():
            out = model.generate_content([prompt, {"path": video_path}])
        return out if isinstance(out, str) else str(out)

    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = build_generate_inputs(
        model, processor, model_type, video_path, question, nframes, fps
    )
    inputs = to_device(inputs, device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return decode_new(processor, inputs, out)


def run_eval(args) -> None:
    # the guard keeps one model per card so parallel runs cannot fight over memory;
    # device_map=auto is the deliberate exception, sharding one model across cards
    n_vis = torch.cuda.device_count()
    if n_vis != 1 and args.device_map != "auto":
        raise SystemExit(
            f"expected 1 visible GPU, got {n_vis} (set CUDA_VISIBLE_DEVICES, "
            "or pass --device-map auto to shard one model over all of them)"
        )
    print(f"GPU: {', '.join(torch.cuda.get_device_name(i) for i in range(n_vis))}", flush=True)

    data_root = Path(args.data_root)
    recs = load_jsonl(Path(args.records))
    if args.limit:
        recs = recs[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path) if args.resume else set()
    todo = [r for r in recs if r.get("video_id") not in done]
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        todo = [r for j, r in enumerate(todo) if j % sn == si]
        print(f"shard {args.shard}: {len(todo)} remaining after skip", flush=True)
    print(f"clips {len(recs)}  skip {len(done)}  todo {len(todo)}  -> {out_path}", flush=True)
    if not todo:
        return

    defaults = REGISTRY.get(args.model_id, {})
    nframes = args.nframes if args.nframes is not None else defaults.get("nframes")
    fps = args.fps if args.fps is not None else defaults.get("fps")
    if args.no_fps:
        fps = None
    nframes = nframes or 16
    print(f"sampling: nframes={nframes} fps={fps}", flush=True)
    max_new = args.max_new_tokens if args.max_new_tokens is not None else defaults.get(
        "max_new_tokens", 32
    )

    model, processor, model_type = load_model(
        args.model_id, args.processor_id, args.device_map
    )
    t0 = time.time()
    n_ok = n_err = 0
    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            rel = rec.get("video") or ""
            path = rel if os.path.isabs(rel) else str(data_root / rel)
            question = args.question or rec.get("question") or QUESTION
            gt = rec.get("count")
            raw = pred = err = None
            t1 = time.time()
            try:
                raw = infer_one(
                    model, processor, model_type, path, question, nframes, fps, max_new
                )
                pred = parse_count(raw)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
            else:
                n_ok += 1
            row = {
                "video_id": rec.get("video_id"),
                "model": args.model_id,
                "pred": pred,
                "gt": gt,
                "k": gt,
                "position": rec.get("position"),
                "phase": rec.get("phase"),
                "parent_video_id": rec.get("parent_video_id"),
                "action_type": rec.get("action_type"),
                "raw_text": raw,
                "error": err,
                "sec": round(time.time() - t1, 3),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i == 1 or i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(todo)}] {rec.get('video_id')} gt={gt} pred={pred} "
                    f"ok={n_ok} err={n_err} {elapsed:.0f}s",
                    flush=True,
                )
    print(f"done ok={n_ok} err={n_err} {time.time() - t0:.0f}s", flush=True)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-smoke", action="store_true")
    ap.add_argument("--per-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--records", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/repcount_nrep")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument(
        "--processor-id",
        default=None,
        help="HF id for AutoProcessor if the model repo has no processor (e.g. Video-R1)",
    )
    ap.add_argument(
        "--device-map",
        default="cuda:0",
        help="'auto' shards the model over every visible GPU, which InternVL needs "
        "at 64 frames; the default keeps one model per GPU",
    )
    ap.add_argument("--nframes", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument(
        "--no-fps",
        action="store_true",
        help="force fixed-frame sampling even when the registry default is fps "
        "(Cosmos defaults to fps=4, which otherwise overrides --nframes)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument(
        "--question",
        default="",
        help="override the count prompt (enumeration ablation, etc.)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/n over remaining clips after resume, e.g. 0/2")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    args = ap.parse_args()

    if args.make_smoke:
        split = args.split if args.split.endswith(".jsonl") else f"{args.split}.jsonl"
        src = Path(args.records) if args.records else Path(args.data_root) / split
        make_smoke(src, args.per_k, args.seed, Path(args.out))
        return
    if not args.records:
        raise SystemExit("--records is required unless --make-smoke")
    run_eval(args)


if __name__ == "__main__":
    main()

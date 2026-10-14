#!/usr/bin/env python3
"""Eval RepNet or ESCounts on RepProbe n-rep clips. Resume-safe JSONL.

Examples:
  CUDA_VISIBLE_DEVICES=0 python eval_repcount_specialists_nrep.py \\
    --model repnet --records data/repcount_nrep/eval_10000.jsonl --resume

  COUNTVID=/path/to/countVid CUDA_VISIBLE_DEVICES=1 \\
    python eval_repcount_specialists_nrep.py --model escounts \\
    --records data/repcount_nrep/eval_10000.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

QUESTION = "Count the number of repetitions in this video. Return only an integer."

COUNTVID = Path(os.environ.get("COUNTVID", ""))
if not COUNTVID.is_dir():
    raise SystemExit("set COUNTVID to a checkout with third_party/RepNet-pytorch and third_party/EveryShotCounts")
REPNET_ROOT = COUNTVID / "third_party" / "RepNet-pytorch"
REPNET_WEIGHTS = REPNET_ROOT / "checkpoints" / "pytorch_weights.pth"
ESCOUNTS_ROOT = COUNTVID / "third_party" / "EveryShotCounts"
ESCOUNTS_WEIGHTS = ESCOUNTS_ROOT / "checkpoints" / "repcount_trained.pyth"
COUNT_GEN = COUNTVID / "count_gen"


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


class RepNetCounter:
    def __init__(self, device: str = "cuda:0", strides: list[int] | None = None):
        import torch
        import torchvision.transforms as T
        from repnet.model import RepNet

        self.torch = torch
        self.device = device
        self.strides = strides or [1, 2, 3, 4, 8]
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
        )
        self.model = RepNet()
        state = torch.load(str(REPNET_WEIGHTS), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval().to(device)

    def count_video(self, video_path: str | Path) -> dict:
        import cv2

        torch = self.torch
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(rgb))
        cap.release()
        n_frames = len(frames)
        if n_frames == 0:
            return {
                "count_pred": None,
                "count_raw": None,
                "confidence": 0.0,
                "stride": None,
                "error": "empty video",
                "n_frames": 0,
            }

        best = None
        with torch.no_grad():
            for stride in self.strides:
                sf = frames[::stride]
                if len(sf) < 64:
                    sf = sf + [sf[-1]] * (64 - len(sf))
                n = (len(sf) // 64) * 64
                sf = sf[:n]
                x = (
                    torch.stack(sf)
                    .unflatten(0, (-1, 64))
                    .movedim(1, 2)
                    .to(self.device)
                )
                pl, ps = [], []
                for i in range(x.shape[0]):
                    a, b, _ = self.model(x[i].unsqueeze(0))
                    pl.append(a[0].cpu())
                    ps.append(b[0].cpu())
                conf, plen, pcount, _ = self.model.get_counts(
                    torch.cat(pl), torch.cat(ps), stride
                )
                conf_f = float(conf)
                raw = float(pcount[-1])
                if best is None or conf_f > best["confidence"]:
                    best = {
                        "count_raw": raw,
                        "count_pred": int(round(raw)),
                        "confidence": conf_f,
                        "stride": stride,
                        "period_length": float(plen),
                        "n_frames": n_frames,
                        "error": None,
                    }
        return best or {
            "count_pred": None,
            "count_raw": None,
            "confidence": 0.0,
            "stride": None,
            "error": "no valid stride",
            "n_frames": n_frames,
        }


class ESCountsCounter:
    def __init__(self, device: str = "cuda:0"):
        if str(COUNT_GEN) not in sys.path:
            sys.path.insert(0, str(COUNT_GEN))
        from count_gen.stage3.escounts_verifier import ESCountsVerifier

        self.v = ESCountsVerifier(weights=ESCOUNTS_WEIGHTS, device=device)

    def count_video(self, video_path: str | Path) -> dict:
        out = self.v.count_video(video_path)
        pred = out.get("count_pred")
        if pred is not None and int(pred) < 0:
            pred = None
        return {
            "count_pred": pred,
            "count_raw": out.get("count_raw"),
            "confidence": out.get("confidence"),
            "n_frames": out.get("n_frames"),
            "error": out.get("error"),
        }

    def close(self) -> None:
        if hasattr(self.v, "close"):
            self.v.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["repnet", "escounts"])
    ap.add_argument("--records", default="data/repcount_nrep/eval_10000.jsonl")
    ap.add_argument("--data-root", default="data/repcount_nrep")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/n e.g. 0/2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.model == "repnet":
        if str(REPNET_ROOT) not in sys.path:
            sys.path.insert(0, str(REPNET_ROOT))
        if not REPNET_WEIGHTS.is_file():
            raise SystemExit(f"missing RepNet weights: {REPNET_WEIGHTS}")
        model_id = "google/RepNet"
        safe = "google_RepNet"
        counter = RepNetCounter(device=args.device)
    else:
        if not ESCOUNTS_WEIGHTS.is_file():
            raise SystemExit(f"missing ESCounts weights: {ESCOUNTS_WEIGHTS}")
        model_id = "EveryShotCounts/ESCounts-RepCount"
        safe = "EveryShotCounts_ESCounts-RepCount"
        counter = ESCountsCounter(device=args.device)

    out_path = Path(
        args.out
        or f"data/repcount_nrep/preds/eval10000__{safe}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = load_jsonl(Path(args.records))
    if args.limit:
        recs = recs[: args.limit]
    done = already_done(out_path) if args.resume else set()
    todo = [r for r in recs if r.get("video_id") not in done]
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        todo = [r for j, r in enumerate(todo) if j % sn == si]
        print(f"shard {args.shard}: {len(todo)} remaining", flush=True)
    print(
        f"model={args.model} clips={len(recs)} skip={len(done)} "
        f"todo={len(todo)} -> {out_path}",
        flush=True,
    )
    if not todo:
        return

    t0 = time.time()
    n_ok = n_err = 0
    data_root = Path(args.data_root)
    with out_path.open("a") as fout:
        for i, rec in enumerate(todo, 1):
            rel = rec.get("video") or ""
            path = Path(rel) if os.path.isabs(rel) else data_root / rel
            gt = rec.get("count")
            t1 = time.time()
            pred = raw = conf = err = None
            extra: dict = {}
            try:
                out = counter.count_video(path)
                pred = out.get("count_pred")
                raw = out.get("count_raw")
                conf = out.get("confidence")
                err = out.get("error")
                for k in ("stride", "period_length", "n_frames"):
                    if k in out:
                        extra[k] = out[k]
                if err:
                    n_err += 1
                else:
                    n_ok += 1
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
            row = {
                "video_id": rec.get("video_id"),
                "model": model_id,
                "pred": pred,
                "gt": gt,
                "k": gt,
                "position": rec.get("position"),
                "action_type": rec.get("action_type"),
                "count_raw": raw,
                "confidence": conf,
                "error": err,
                "sec": round(time.time() - t1, 3),
                **extra,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(todo):
                print(
                    f"[{i}/{len(todo)}] {rec.get('video_id')} "
                    f"gt={gt} pred={pred} ok={n_ok} err={n_err} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

    if hasattr(counter, "close"):
        counter.close()
    print(f"done ok={n_ok} err={n_err} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

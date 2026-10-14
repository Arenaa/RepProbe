#!/usr/bin/env python3
"""Build a RepCount n-rep dataset from per-cycle annotations.

Cuts contiguous k-cycle windows at annotated rest-pose boundaries (never
mid-rep). For each source video and each k, keep start / middle / end
windows so a 3-rep clip is not always the first three cycles.

Input:
  HuggingFace lmms-lab-eval/repcount metadata (question text)
  Official RepCount CSVs with L1,L2,... frame bounds per cycle
  Official RepCount videos

Output (under --out-dir):
  count_{k}.jsonl              primary splits by repetition count
  {train,validation,test}.jsonl  same rows grouped by source split
  videos/count_{k}/*.mp4
  source_annotations/*.csv
  summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

QUESTION = "Count the number of repetitions in this video. Return only an integer."

SPLIT_CSV = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}
SPLIT_VIDEO = {"train": "train", "valid": "valid", "test": "test"}
SPLIT_HF = {"train": "train", "valid": "validation", "test": "test"}
HF_JSONL = {
    "train": "train.jsonl",
    "valid": "validation.jsonl",
    "test": "test.jsonl",
}


@dataclass(frozen=True)
class Cycle:
    start: int  # inclusive frame
    end: int    # exclusive rest-pose boundary (next cycle start, or last L)


@dataclass
class Window:
    source_name: str
    source_split: str  # train | valid | test
    action_type: str
    hf_action_type: Optional[str]
    source_count: int
    source_n_cycles: int
    fps: float
    num_frames: int
    video_path: str
    count: int
    position: str  # start | middle | end
    cycle_indices: tuple[int, ...]
    src_start_frame: int
    src_end_frame: int

    @property
    def clip_id(self) -> str:
        stem = Path(self.source_name).stem
        return f"{stem}__n{self.count}__{self.position}"

    @property
    def rel_video(self) -> str:
        return f"videos/count_{self.count}/{self.clip_id}.mp4"

    @property
    def src_start_sec(self) -> float:
        return self.src_start_frame / self.fps

    @property
    def src_end_sec(self) -> float:
        return self.src_end_frame / self.fps


def parse_cycles(row: dict[str, str]) -> list[Cycle]:
    """Parse L1,L2,... as (start,end) pairs. Drop zero-length periods.

    RepCount stores each cycle as consecutive L columns; adjacent cycles
    share the rest-pose frame (end_i == start_{i+1}). We treat the bound
    as a half-open interval [start, end) so windows never bleed into the
    next cycle's motion.
    """
    bounds: list[int] = []
    for i in range(1, 400):
        raw = (row.get(f"L{i}") or "").strip()
        if not raw:
            break
        try:
            bounds.append(int(float(raw)))
        except ValueError:
            break

    cycles: list[Cycle] = []
    last_start = -1
    for k in range(0, len(bounds) - 1, 2):
        s, e = bounds[k], bounds[k + 1]
        if e <= s:
            continue
        if s < last_start:
            # Non-monotonic annotation: unusable.
            return []
        last_start = s
        cycles.append(Cycle(s, e))
    return cycles


def window_offsets(n_cycles: int, k: int) -> list[tuple[str, int]]:
    """Start / middle / end offsets for a contiguous k-cycle window."""
    if k < 1 or n_cycles < k:
        return []
    start, end = 0, n_cycles - k
    mid = end // 2
    out = [("start", start)]
    if mid != start:
        out.append(("middle", mid))
    if end not in (start, mid):
        out.append(("end", end))
    return out


def load_hf_meta(hf_dir: Path) -> dict[str, dict]:
    """Map source filename -> HF row (question, action_type, original count)."""
    out: dict[str, dict] = {}
    for split, fname in HF_JSONL.items():
        path = hf_dir / fname
        if not path.is_file():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("video") or f"{rec.get('video_id')}.mp4"
                rec["_hf_split"] = SPLIT_HF[split]
                out[key] = rec
    return out


def load_source_videos(
    ann_dir: Path,
    video_root: Path,
    hf_meta: dict[str, dict],
) -> tuple[list[dict], Counter]:
    skipped: Counter = Counter()
    videos: list[dict] = []
    for split, csv_name in SPLIT_CSV.items():
        path = ann_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name:
                    skipped["empty_name"] += 1
                    continue
                vpath = video_root / SPLIT_VIDEO[split] / name
                if not vpath.is_file():
                    skipped["missing_video"] += 1
                    continue
                cycles = parse_cycles(row)
                if not cycles:
                    skipped["no_valid_cycles"] += 1
                    continue
                try:
                    fps = float(row.get("fps") or 0) or 30.0
                except ValueError:
                    fps = 30.0
                try:
                    num_frames = int(float(row.get("num_frames") or 0))
                except ValueError:
                    num_frames = cycles[-1].end
                try:
                    source_count = int(float(row.get("count") or len(cycles)))
                except ValueError:
                    source_count = len(cycles)
                hf = hf_meta.get(name, {})
                videos.append(
                    {
                        "name": name,
                        "split": split,
                        "action_type": (row.get("type") or hf.get("action_type") or "unknown"),
                        "hf_action_type": hf.get("action_type"),
                        "source_count": source_count,
                        "n_cycles": len(cycles),
                        "cycles": cycles,
                        "fps": fps,
                        "num_frames": num_frames,
                        "video_path": str(vpath),
                    }
                )
    return videos, skipped


def enumerate_windows(videos: list[dict], counts: tuple[int, ...]) -> list[Window]:
    windows: list[Window] = []
    for v in videos:
        cycles: list[Cycle] = v["cycles"]
        n = len(cycles)
        max_end = v["num_frames"] if v["num_frames"] > 0 else cycles[-1].end
        for k in counts:
            for position, offset in window_offsets(n, k):
                idxs = tuple(range(offset, offset + k))
                start_f = cycles[offset].start
                end_f = cycles[offset + k - 1].end
                if end_f <= start_f:
                    continue
                # Never read past the annotated video length.
                if start_f >= max_end:
                    continue
                end_f = min(end_f, max_end)
                if end_f - start_f < 2:
                    continue
                windows.append(
                    Window(
                        source_name=v["name"],
                        source_split=v["split"],
                        action_type=v["action_type"],
                        hf_action_type=v["hf_action_type"],
                        source_count=v["source_count"],
                        source_n_cycles=n,
                        fps=v["fps"],
                        num_frames=v["num_frames"],
                        video_path=v["video_path"],
                        count=k,
                        position=position,
                        cycle_indices=idxs,
                        src_start_frame=start_f,
                        src_end_frame=end_f,
                    )
                )
    return windows


def _cut_one(args: tuple[str, str, int, int]) -> tuple[str, bool, str]:
    src, dst, start_f, end_f = args
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst, True, "exists"
    vf = f"select=gte(n\\,{start_f})*lt(n\\,{end_f}),setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-vf", vf,
        "-an",
        "-vsync", "0",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "replace")[:400]
        if os.path.isfile(dst):
            os.remove(dst)
        return dst, False, err or "ffmpeg failed"
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        return dst, False, "empty output"
    return dst, True, "ok"


def window_record(w: Window) -> dict:
    return {
        "video": w.rel_video,
        "video_id": w.clip_id,
        "count": w.count,
        "question": QUESTION,
        "action_type": w.hf_action_type or w.action_type,
        "raw_source": "RepCount",
        "source_video": w.source_name,
        "source_split": SPLIT_HF[w.source_split],
        "source_count": w.source_count,
        "source_n_cycles": w.source_n_cycles,
        "position": w.position,
        "cycle_indices": list(w.cycle_indices),
        "src_start_frame": w.src_start_frame,
        "src_end_frame": w.src_end_frame,
        "src_start_sec": round(w.src_start_sec, 4),
        "src_end_sec": round(w.src_end_sec, 4),
        "fps": w.fps,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", default="data/repcount")
    ap.add_argument(
        "--ann-dir",
        default=os.environ.get("REPCOUNT_ANN", ""),
        help="RepCount CSVs with L1,L2,... cycle bounds (or set REPCOUNT_ANN)",
    )
    ap.add_argument(
        "--video-root",
        default=os.environ.get("REPCOUNT_VIDEO", ""),
        help="Root with train/ valid/ test/ video folders (or set REPCOUNT_VIDEO)",
    )
    ap.add_argument("--out-dir", default="data/repcount_nrep")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--max-count", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--metadata-only",
        action="store_true",
        help="rewrite JSONL for windows whose videos already exist; do not run ffmpeg",
    )
    args = ap.parse_args()
    if not args.ann_dir or not args.video_root:
        raise SystemExit("pass --ann-dir and --video-root (or set REPCOUNT_ANN / REPCOUNT_VIDEO)")

    hf_dir = Path(args.hf_dir)
    ann_dir = Path(args.ann_dir)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    counts = tuple(range(args.min_count, args.max_count + 1))

    hf_meta = load_hf_meta(hf_dir)
    videos, skipped = load_source_videos(ann_dir, video_root, hf_meta)
    windows = enumerate_windows(videos, counts)
    print(
        f"sources={len(videos)} skipped={dict(skipped)} "
        f"windows={len(windows)} counts={counts}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    src_ann = out_dir / "source_annotations"
    src_ann.mkdir(exist_ok=True)
    for csv_name in SPLIT_CSV.values():
        src = ann_dir / csv_name
        if src.is_file():
            shutil.copy2(src, src_ann / csv_name)

    if args.dry_run:
        jobs = windows
        failed: list[str] = []
    elif args.metadata_only:
        jobs = [w for w in windows if (out_dir / w.rel_video).is_file()]
        failed = []
        print(f"metadata-only: {len(jobs)}/{len(windows)} videos present")
    else:
        jobs = []
        failed = []
        work = [
            (
                w.video_path,
                str(out_dir / w.rel_video),
                w.src_start_frame,
                w.src_end_frame,
            )
            for w in windows
        ]
        ok_dst = set()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_cut_one, job): i for i, job in enumerate(work)}
            n = len(futs)
            done = 0
            for fut in as_completed(futs):
                dst, ok, msg = fut.result()
                done += 1
                if ok:
                    ok_dst.add(dst)
                else:
                    failed.append(f"{dst}: {msg}")
                if done % 500 == 0 or done == n:
                    print(f"  cut {done}/{n}  failed={len(failed)}")
        jobs = [
            w for w in windows
            if str(out_dir / w.rel_video) in ok_dst
        ]

    records = [window_record(w) for w in jobs]
    by_count: dict[int, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    pos_hist: Counter = Counter()
    for rec in records:
        by_count[rec["count"]].append(rec)
        by_split[rec["source_split"]].append(rec)
        pos_hist[(rec["count"], rec["position"])] += 1

    write_jsonl(out_dir / "all.jsonl", records)
    for k in counts:
        write_jsonl(out_dir / f"count_{k}.jsonl", by_count.get(k, []))
    for split in ("train", "validation", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", by_split.get(split, []))

    summary = {
        "description": (
            "RepCount subclips of exactly k contiguous annotated cycles, "
            "cut at rest-pose boundaries. Splits are by k (count_1..count_N); "
            "each k includes start/middle/end windows when the source has "
            "enough cycles."
        ),
        "n_source_videos": len(videos),
        "n_clips": len(records),
        "counts": list(counts),
        "count_hist": {str(k): len(by_count.get(k, [])) for k in counts},
        "source_split_hist": {s: len(by_split.get(s, [])) for s in ("train", "validation", "test")},
        "position_hist": {
            f"count_{k}_{p}": pos_hist[(k, p)]
            for k in counts
            for p in ("start", "middle", "end")
        },
        "skipped_sources": dict(skipped),
        "n_failed_cuts": len(failed),
        "dry_run": bool(args.dry_run),
        "metadata_only": bool(args.metadata_only),
        "hf_dir": str(hf_dir.resolve()),
        "ann_dir": str(ann_dir.resolve()),
        "video_root": str(video_root.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failed:
        (out_dir / "failed_cuts.txt").write_text("\n".join(failed) + "\n")

    print(f"wrote {len(records)} clips -> {out_dir}")
    for k in counts:
        print(f"  count_{k}: {len(by_count.get(k, []))}")
    if failed:
        print(f"  failed cuts: {len(failed)} (see failed_cuts.txt)")


if __name__ == "__main__":
    main()

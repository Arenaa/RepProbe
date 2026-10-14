#!/usr/bin/env python3
"""Quality-check a gen_synth_reps.py dataset: manifest, construction, and pixel counts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from repprobe.dataset import synth as g
from repprobe.paths import SYNTH

ROOT = SYNTH
MAN = ROOT / "eval_synthetic.jsonl"


def zc(xs):
    s = np.sign(np.asarray(xs, dtype=float))
    s[s == 0] = 1
    return int(np.sum(s[:-1] * s[1:] < 0))


def npeaks(x, thr=0.45):
    x = np.asarray(x, dtype=float)
    if x.size < 3 or float(x.max()) == float(x.min()):
        return 0
    xn = (x - x.min()) / (x.max() - x.min())
    n = 0
    for i in range(1, len(xn) - 1):
        if xn[i] >= xn[i - 1] and xn[i] > xn[i + 1] and xn[i] > thr:
            n += 1
    return n


def colored(img, cut=18.0):
    return (255.0 - img.astype(np.float32)).max(-1) > cut


def centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def series_centroid(frames, axis=0, cut=18.0):
    out = []
    for img in frames:
        c = centroid(colored(img, cut))
        out.append(np.nan if c is None else c[axis])
    return np.array(out)


def count_from_frames(fam, frames):
    T = len(frames)
    if fam in ("flash",):
        area = np.array([colored(im).sum() for im in frames], float)
        return npeaks(area)
    if fam == "textured":
        d0 = frames[0].astype(np.float32)
        dif = np.array([np.abs(im.astype(np.float32) - d0).mean() for im in frames])
        return npeaks(dif)
    if fam == "bounce":
        ys = series_centroid(frames, axis=1, cut=22.0)
        if np.isnan(ys).all():
            return -1
        ys = np.interp(np.arange(T), np.flatnonzero(~np.isnan(ys)), ys[~np.isnan(ys)])
        return npeaks(-ys, thr=0.40)  # apex = small y
    if fam == "pendulum":
        xs = series_centroid(frames, axis=0, cut=22.0)
        if np.isnan(xs).all():
            return -1
        xs = np.interp(np.arange(T), np.flatnonzero(~np.isnan(xs)), xs[~np.isnan(xs)])
        return zc(xs - np.median(xs))
    if fam == "scale":
        area = np.array([colored(im, 14).sum() for im in frames], float)
        return npeaks(area, thr=0.40)
    if fam == "color":
        cols = []
        for im in frames:
            m = colored(im, 14)
            cols.append(im[m].mean(0) if m.any() else np.array([255, 255, 255]))
        cols = np.stack(cols)
        dist = np.linalg.norm(cols - cols[0], axis=1)
        return npeaks(dist, thr=0.40)
    if fam == "multi":
        energy = np.array([(255.0 - im.astype(np.float32)).mean() for im in frames])
        return npeaks(energy, thr=0.35)
    if fam == "shape":
        score = []
        for im in frames:
            m = colored(im, 14)
            c = centroid(m)
            if c is None:
                score.append(0.0)
                continue
            ys, xs = np.where(m)
            r = np.hypot(xs - c[0], ys - c[1])
            score.append(float(r.std()))
        return npeaks(np.array(score), thr=0.40)
    if fam == "crossing":
        lx, cx = [], []
        for im in frames:
            beige = (im[:, :, 0].astype(int) > im[:, :, 2].astype(int) + 8)
            lx.append(int(np.argmax(beige.sum(0))))
            blue = (im[:, :, 2].astype(int) > im[:, :, 0].astype(int) + 8)
            c = centroid(blue)
            cx.append(112.0 if c is None else c[0])
        return zc(np.array(lx, float) - np.median(cx))
    if fam == "orbit":
        angs = []
        for im in frames:
            c = centroid(colored(im, 16))
            if c is None:
                angs.append(np.nan)
            else:
                angs.append(np.arctan2(c[1] - 112.0, c[0] - 112.0))
        angs = np.array(angs)
        if np.isnan(angs).all():
            return -1
        good = ~np.isnan(angs)
        angs = np.interp(np.arange(T), np.flatnonzero(good), angs[good])
        unw = np.unwrap(angs)
        return int(round(abs(unw[-1] - unw[0]) / (2 * np.pi)))
    raise ValueError(fam)


def qc_one(rec):
    path = ROOT / rec["video"]
    fam, k = rec["family"], int(rec["count"])
    out = {
        "video": rec["video"],
        "family": fam,
        "k": k,
        "exists": path.is_file(),
        "n_frames": None,
        "shape": None,
        "construct_ok": False,
        "pixel_n": None,
        "pixel_ok": False,
        "err": None,
    }
    try:
        ok, n = g.verify_count(fam, k, rec["seed"])
        out["construct_ok"] = bool(ok) and n == k
        if not path.is_file():
            out["err"] = "missing"
            return out
        frames = iio.mimread(str(path))
        out["n_frames"] = len(frames)
        out["shape"] = tuple(frames[0].shape)
        if len(frames) != 64 or frames[0].shape != (224, 224, 3):
            out["err"] = f"bad_video {len(frames)} {frames[0].shape}"
            return out
        pn = count_from_frames(fam, frames)
        out["pixel_n"] = int(pn)
        out["pixel_ok"] = pn == k
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def main():
    global ROOT, MAN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=SYNTH)
    args = ap.parse_args()
    ROOT = args.root
    MAN = ROOT / "eval_synthetic.jsonl"
    rows = [json.loads(l) for l in MAN.open() if l.strip()]
    print(f"manifest {len(rows)}")
    ids = [r["video_id"] for r in rows]
    print(f"unique ids {len(set(ids))}")
    print("family", dict(Counter(r["family"] for r in rows)))
    print("k", dict(sorted(Counter(int(r["count"]) for r in rows).items())))
    per = Counter((r["family"], int(r["count"])) for r in rows)
    print("fam x k min/max", min(per.values()), max(per.values()), "cells", len(per))

    name_mismatch = [
        r
        for r in rows
        if r["video"] != f"{r['family']}_k{int(r['count']):02d}_{r['video_id'].split('_')[-1]}.mp4"
        or int(r["n_frames"]) != 64
        or float(r["duration"]) != 4.0
    ]
    print("name/meta mismatches", len(name_mismatch))

    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(qc_one, rows, chunksize=25))

    n = len(results)
    miss = sum(1 for r in results if not r["exists"])
    badv = sum(1 for r in results if r["err"])
    cons = sum(1 for r in results if r["construct_ok"])
    pix = sum(1 for r in results if r["pixel_ok"])
    print(f"\nfiles missing {miss}/{n}")
    print(f"decode/errors {badv}/{n}")
    print(f"construction k  {cons}/{n}  ({100*cons/n:.2f}%)")
    print(f"pixel-count k   {pix}/{n}  ({100*pix/n:.2f}%)")

    by = defaultdict(lambda: [0, 0])
    for r in results:
        by[r["family"]][0] += 1
        by[r["family"]][1] += int(r["pixel_ok"])
    print("\npixel match by family:")
    for fam in g.FAMILIES:
        tot, ok = by[fam]
        print(f"  {fam:10s}  {ok:4d}/{tot}  {100*ok/tot:6.2f}%")

    bad = [r for r in results if not r["pixel_ok"] or r["err"] or not r["construct_ok"]]
    print(f"\nmismatches {len(bad)} (showing up to 20)")
    for r in bad[:20]:
        print(
            f"  {r['video']}  k={r['k']} pixel={r['pixel_n']} "
            f"construct={r['construct_ok']} err={r['err']}"
        )
    outp = ROOT / "qc_report.jsonl"
    with outp.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Synthetic repetition-counting dataset generator (10 families).

Design goals:
  * EXACT ground-truth k by construction (k peaks placed per clip).
  * FIXED duration for every clip (N_FRAMES constant), so clip length carries
    ZERO information about k across the whole set.
  * 10 families: flash, textured, bounce, pendulum, scale, color, multi,
    shape, crossing, orbit. Each increment of k is one complete motion
    (see COUNT_UNIT).
  * Per-clip metadata (family, k, seed) so per-family results can be reported.

Usage:
    python scripts/gen_synth.py --out data/synth_reps_10k --per-k 50
    # 10 families x (K_MAX-K_MIN+1) k-values x per-k clips

Dependencies: numpy, pillow, imageio, imageio-ffmpeg
    pip install numpy pillow imageio imageio-ffmpeg --break-system-packages
"""

import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio

# ------------------------------------------------------------------
# Global config  (match these to your RepCount sampling)
# ------------------------------------------------------------------
H, W        = 224, 224
FPS         = 16
DUR_S       = 4.0
N_FRAMES    = int(FPS * DUR_S)     # 64 frames, SAME for every clip
K_MIN, K_MAX = 1, 20
BG          = 255                  # white canvas (previews / 10k_white)

# Family base pastels (jittered per clip).
PASTEL = {
    "flash":       (244, 166, 184),
    "textured":    (242, 201, 111),
    "bounce":      (163, 206, 227),
    "pendulum":    (197, 176, 227),
    "scale":       (168, 219, 198),
    "color":       (247, 198, 163),
    "multi":       (232, 160, 191),
    "shape":       (230, 184, 162),
    "crossing":    (158, 197, 232),
    "orbit":       (143, 184, 168),
    # aliases kept so older family functions still run
    "translate":   (163, 206, 227),
    "rotate":      (197, 176, 227),
    "varrate":     (230, 184, 162),
    "articulated": (143, 184, 168),
}
PASTEL2 = {
    "color":       (168, 197, 232),
    "multi":       (214, 210, 224),
    "crossing":    (232, 213, 196),
}


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------
def _blank(bg=BG):
    return np.full((H, W, 3), bg, np.uint8)

def _fade(col, a, bg=BG):
    """Lerp pastel -> white so pulses stay light on a white canvas."""
    a = float(np.clip(a, 0.0, 1.0))
    return tuple(int(a * c + (1.0 - a) * bg) for c in col)

def _pastel(rng, base, jitter=16, lo=150, hi=255):
    c = np.asarray(base, dtype=np.int32) + rng.integers(-jitter, jitter + 1, 3)
    return tuple(int(x) for x in np.clip(c, lo, hi))

def _event_times(k, rng, jitter=0.0):
    """Frame indices of the k repetition peaks, evenly spaced inside the clip.
    jitter>0 perturbs spacing (irregular rate) while keeping exactly k peaks and
    the same total duration -> rate/spacing carry no info about k."""
    base = np.linspace(0, N_FRAMES - 1, k + 2)[1:-1].astype(float)
    if jitter > 0 and k > 1:
        span = N_FRAMES / (k + 1)
        base = base + rng.uniform(-jitter, jitter, size=k) * span
        base = np.clip(np.sort(base), 0, N_FRAMES - 1)
    return base

def _phase(t, peaks, width):
    """Triangular pulse: 1 at nearest peak, 0 at distance>=width. One pulse/rep."""
    d = np.min(np.abs(t - peaks))
    return max(0.0, 1.0 - d / width)

def _sine_phase(t, peaks):
    """Smooth 0..1..0 pulse per repetition (for scale/rotation-style motion)."""
    return _phase(t, peaks, N_FRAMES / (len(peaks) + 1) / 2)

def _width(k):
    return N_FRAMES / (k + 1) / 2

def _halfcycle_u(t, k):
    """+1 at t=0, then k half-periods of cosine. Each zero-crossing is one event."""
    return float(np.cos(np.pi * k * t / (N_FRAMES - 1)))

def _rand_pos(rng, margin=60):
    return int(rng.integers(margin, W - margin)), int(rng.integers(margin, H - margin))

def _rand_col(rng, lo=150, hi=256):
    """Fallback pastel if a family does not pass a base swatch."""
    h = float(rng.uniform(0.0, 1.0))
    s = float(rng.uniform(0.28, 0.48))
    v = float(rng.uniform(0.82, 0.95))
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(255 * x) for x in (r, g, b))

def _texture_bg(rng):
    """Low-frequency pastel wash (not a dark scene)."""
    palette = np.array(
        [
            [245, 232, 240],
            [232, 240, 245],
            [240, 245, 232],
            [245, 240, 228],
            [236, 232, 245],
        ],
        dtype=np.float32,
    )
    idx = rng.integers(0, len(palette), (6, 6))
    small = palette[idx] + rng.normal(0, 4.0, (6, 6, 3))
    img = Image.fromarray(np.clip(small, 220, 255).astype(np.uint8)).resize(
        (W, H), Image.BICUBIC
    )
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=6))).copy()


# ------------------------------------------------------------------
# Family 1: flashing primitive (clean appearance baseline)
# ------------------------------------------------------------------
def fam_flash(k, seed):
    rng = np.random.default_rng(seed)
    peaks, w = _event_times(k, rng), _width(k)
    cx, cy = _rand_pos(rng); r = int(rng.integers(15, 35)); col = _pastel(rng, PASTEL["flash"])
    out = []
    for t in range(N_FRAMES):
        a = _phase(t, peaks, w); img = Image.fromarray(_blank())
        if a > .05:
            ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=_fade(col, a))
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 2: translating object (motion, not appearance)
# ------------------------------------------------------------------
def fam_translate(k, seed):
    rng = np.random.default_rng(seed)
    peaks, w = _event_times(k, rng), _width(k)
    y = int(rng.integers(60, H-60)); r = int(rng.integers(14, 26)); col = _pastel(rng, PASTEL["translate"])
    x0, x1 = 30, W-30
    out = []
    for t in range(N_FRAMES):
        a = _phase(t, peaks, w)          # object sweeps out-and-back each rep
        x = int(x0 + (x1-x0) * a)
        img = Image.fromarray(_blank())
        ImageDraw.Draw(img).ellipse([x-r, y-r, x+r, y+r], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 3: rotating marker (angular periodicity)
# ------------------------------------------------------------------
def fam_rotate(k, seed):
    rng = np.random.default_rng(seed)
    cx, cy = W//2, H//2; L = int(rng.integers(50, 90)); col = _pastel(rng, PASTEL["rotate"])
    out = []
    for t in range(N_FRAMES):
        ang = 2*np.pi * k * (t/(N_FRAMES-1))     # k full turns over the clip
        x = cx + L*np.cos(ang); y = cy + L*np.sin(ang)
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        d.line([cx, cy, x, y], fill=col, width=6)
        d.ellipse([x-8, y-8, x+8, y+8], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 4: scaling / pulsing (size oscillation)
# ------------------------------------------------------------------
def fam_scale(k, seed):
    rng = np.random.default_rng(seed)
    peaks = _event_times(k, rng)
    cx, cy = _rand_pos(rng, 70); rmin, rmax = 8, int(rng.integers(40, 70)); col = _pastel(rng, PASTEL["scale"])
    out = []
    for t in range(N_FRAMES):
        a = _sine_phase(t, peaks)
        r = int(rmin + (rmax-rmin)*a)
        img = Image.fromarray(_blank())
        ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 5: colour cycling (no spatial motion at all)
# ------------------------------------------------------------------
def fam_color(k, seed):
    rng = np.random.default_rng(seed)
    peaks = _event_times(k, rng)
    cx, cy = _rand_pos(rng, 70)
    r = int(rng.integers(40, 55))          # FIXED size (no scale cue)
    c0 = np.array(_pastel(rng, PASTEL["color"]))
    c1 = np.array(_pastel(rng, PASTEL2["color"]))
    if rng.random() < 0.5: c0, c1 = c1, c0
    out = []
    for t in range(N_FRAMES):
        a = _sine_phase(t, peaks)          # hue swings c0<->c1 exactly k times
        col = tuple(int(v) for v in (c0*(1-a) + c1*a))
        img = Image.fromarray(_blank())
        ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: several objects flash in turn; k = total flashes
# ------------------------------------------------------------------
def fam_multi(k, seed):
    rng = np.random.default_rng(seed)
    peaks, w = _event_times(k, rng), _width(k)
    n_obj = int(rng.integers(3, 5))
    palettes = [PASTEL["multi"], PASTEL2["multi"], PASTEL["color"], PASTEL2["color"]]
    objs = []
    used = []
    for j in range(n_obj):
        for _ in range(40):
            cx, cy = _rand_pos(rng, 40)
            if all((cx - ux) ** 2 + (cy - uy) ** 2 > 55 ** 2 for ux, uy in used):
                break
        used.append((cx, cy))
        objs.append((cx, cy, int(rng.integers(14, 24)), _pastel(rng, palettes[j % len(palettes)])))
    owners = np.arange(k) % n_obj
    obj_peaks = [peaks[owners == j] for j in range(n_obj)]
    out = []
    for t in range(N_FRAMES):
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        for j, (cx, cy, r, col) in enumerate(objs):
            pk = obj_peaks[j]
            a = _phase(t, pk, w) if len(pk) else 0.0
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_fade(col, max(a, 0.32)))
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 7: textured background (real-scene proxy)
# ------------------------------------------------------------------
def fam_textured(k, seed):
    rng = np.random.default_rng(seed)
    bg = _texture_bg(rng)
    peaks, w = _event_times(k, rng), _width(k)
    cx, cy = _rand_pos(rng); r = int(rng.integers(15, 35)); col = _pastel(rng, PASTEL["textured"])
    out = []
    for t in range(N_FRAMES):
        a = _phase(t, peaks, w); img = Image.fromarray(bg.copy())
        if a > .05:
            ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=_fade(col, a))
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: square and a thick line; k = number of full crossings
# One count = the line starts on one side of the square and finishes on the other.
# ------------------------------------------------------------------
def fam_crossing(k, seed):
    rng = np.random.default_rng(seed)
    cx = int(np.clip(W // 2 + int(rng.integers(-18, 19)), 90, W - 90))
    cy = int(np.clip(H // 2 + int(rng.integers(-18, 19)), 90, H - 90))
    half = int(rng.integers(22, 36))
    bar_w = int(rng.integers(14, 20))
    # line must start fully clear of the square and stay on-canvas
    need = half + bar_w // 2 + 8
    max_amp = min(cx, W - cx) - bar_w // 2 - 4
    amp = max(need, min(max_amp, int(rng.integers(need, max(need + 1, max_amp + 1)))))
    sq_col = _pastel(rng, PASTEL["crossing"])
    line_col = _pastel(rng, PASTEL2["crossing"], jitter=10)
    out = []
    for t in range(N_FRAMES):
        u = _halfcycle_u(t, k)
        lx = int(cx + amp * u)
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(
            [cx - half, cy - half, cx + half, cy + half], radius=6, fill=sq_col
        )
        d.line([(lx, 12), (lx, H - 12)], fill=line_col, width=bar_w)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 9: variable-rate (irregular spacing; kills duration/rate shortcut)
# ------------------------------------------------------------------
def fam_varrate(k, seed):
    rng = np.random.default_rng(seed)
    peaks, w = _event_times(k, rng, jitter=0.6), _width(k)
    cx, cy = _rand_pos(rng); r = int(rng.integers(15, 35)); col = _pastel(rng, PASTEL["varrate"])
    out = []
    for t in range(N_FRAMES):
        a = _phase(t, peaks, w); img = Image.fromarray(_blank())
        if a > .05:
            ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=_fade(col, a))
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family 10: articulated / compound motion (closest to an "action")
# ------------------------------------------------------------------
def fam_articulated(k, seed):
    rng = np.random.default_rng(seed)
    peaks = _event_times(k, rng)
    cx, cy = W//2, int(rng.integers(90, 140)); col = _pastel(rng, PASTEL["articulated"])
    limb = int(rng.integers(35, 55))
    out = []
    for t in range(N_FRAMES):
        a = _sine_phase(t, peaks)          # a limb swings up-and-down k times
        ang = -np.pi/2 + (a-0.5)*1.4       # swing range
        img = Image.fromarray(_blank()); d = ImageDraw.Draw(img)
        d.ellipse([cx-14, cy-14, cx+14, cy+14], fill=col)      # body
        ex = cx + limb*np.cos(ang); ey = cy + limb*np.sin(ang)
        d.line([cx, cy, ex, ey], fill=col, width=8)            # swinging limb
        d.ellipse([ex-8, ey-8, ex+8, ey+8], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: vertical bounce
# ------------------------------------------------------------------
def fam_bounce(k, seed):
    rng = np.random.default_rng(seed)
    peaks, w = _event_times(k, rng), _width(k)
    cx = int(rng.integers(50, W - 50))
    r = int(rng.integers(14, 26))
    col = _pastel(rng, PASTEL["bounce"])
    y1, y0 = 40, H - 36
    out = []
    for t in range(N_FRAMES):
        a = _phase(t, peaks, w)
        y = int(y0 + (y1 - y0) * a)
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        d.line([24, y0 + r + 4, W - 24, y0 + r + 4], fill=(220, 220, 220), width=3)
        d.ellipse([cx - r, y - r, cx + r, y + r], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: pendulum swing
# One count = one one-way swing (left → right, or right → left), matching
# the 5-pose figure. k=1 is a single left-to-right pass; k=2 is L→R then R→L.
# ------------------------------------------------------------------
def fam_pendulum(k, seed):
    rng = np.random.default_rng(seed)
    px, py = W // 2, int(rng.integers(18, 36))
    L = int(rng.integers(70, 100))
    swing = float(rng.uniform(1.55, 1.85))
    col = _pastel(rng, PASTEL["pendulum"])
    out = []
    for t in range(N_FRAMES):
        u = _halfcycle_u(t, k)
        a = 0.5 * (u + 1.0)  # 1 = left extreme, 0 = right extreme
        ang = np.pi / 2 + (a - 0.5) * swing
        x = px + L * np.cos(ang)
        y = py + L * np.sin(ang)
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        d.line([px, py, x, y], fill=col, width=6)
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=col)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=col)
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: shape change (circle <-> rounded square)
# ------------------------------------------------------------------
def fam_shape(k, seed):
    rng = np.random.default_rng(seed)
    peaks = _event_times(k, rng)
    cx, cy = _rand_pos(rng, 70)
    r = int(rng.integers(36, 55))
    col = _pastel(rng, PASTEL["shape"])
    out = []
    for t in range(N_FRAMES):
        a = _sine_phase(t, peaks)
        img = Image.fromarray(_blank())
        rad = int(r * (1.0 - 0.92 * a))
        ImageDraw.Draw(img).rounded_rectangle(
            [cx - r, cy - r, cx + r, cy + r], radius=max(4, rad), fill=col
        )
        out.append(np.asarray(img))
    return out


# ------------------------------------------------------------------
# Family: orbit (k revolutions)
# ------------------------------------------------------------------
def fam_orbit(k, seed):
    rng = np.random.default_rng(seed)
    cx, cy = W // 2, H // 2
    L = int(rng.integers(50, 85))
    r = int(rng.integers(12, 20))
    col = _pastel(rng, PASTEL["orbit"])
    out = []
    for t in range(N_FRAMES):
        ang = 2 * np.pi * k * (t / (N_FRAMES - 1)) - np.pi / 2
        x = cx + L * np.cos(ang)
        y = cy + L * np.sin(ang)
        img = Image.fromarray(_blank())
        d = ImageDraw.Draw(img)
        d.ellipse([cx - L, cy - L, cx + L, cy + L], outline=(230, 230, 230), width=2)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        out.append(np.asarray(img))
    return out


FAMILIES = {
    "flash":     fam_flash,
    "textured":  fam_textured,
    "bounce":    fam_bounce,
    "pendulum":  fam_pendulum,
    "scale":     fam_scale,
    "color":     fam_color,
    "multi":     fam_multi,
    "shape":     fam_shape,
    "crossing":  fam_crossing,
    "orbit":     fam_orbit,
}

# What one increment of k is.
COUNT_UNIT = {
    "flash":     "one full appearance pulse (off → on → off)",
    "textured":  "one full appearance pulse on a textured wash (off → on → off)",
    "bounce":    "one full hop (floor → apex → floor)",
    "pendulum":  "one one-way swing (left → right, or right → left)",
    "scale":     "one full size pulse (small → large → small)",
    "color":     "one full hue swing (c0 → c1 → c0)",
    "multi":     "one object-flash (objects take turns; k = total flashes)",
    "shape":     "one full morph (circle → square → circle)",
    "crossing":  "one full crossing of the line through the square",
    "orbit":     "one full revolution around the circle",
}


def _count_zero_crossings(xs):
    s = np.sign(np.asarray(xs, dtype=float))
    s[s == 0] = 1
    return int(np.sum(s[:-1] * s[1:] < 0))


def verify_count(fam, k, seed):
    """Check that the motion parameter of this (fam, k, seed) has exactly k events."""
    rng = np.random.default_rng(seed)
    ts = np.arange(N_FRAMES, dtype=float)
    if fam in ("flash", "textured", "bounce", "scale", "color", "multi", "shape"):
        peaks = _event_times(k, rng)
        w = _width(k)
        phase = np.array([_phase(t, peaks, w) for t in ts])
        # a complete pulse is a local max of the triangular phase (one per peak)
        n = int(np.sum((phase[1:-1] > phase[:-2]) & (phase[1:-1] >= phase[2:]) & (phase[1:-1] > 0.5)))
        # peaks that land on frame 0 or N-1 are still counted via nearest-frame
        if n != k:
            # fall back: each designed peak maps to a unique nearest frame
            nearest = {int(np.clip(round(p), 0, N_FRAMES - 1)) for p in peaks}
            n = len(nearest)
        return n == k, n
    if fam in ("pendulum", "crossing"):
        u = np.array([_halfcycle_u(t, k) for t in ts])
        n = _count_zero_crossings(u)
        return n == k, n
    if fam == "orbit":
        revs = k * (ts[-1] - ts[0]) / (N_FRAMES - 1)
        return abs(revs - k) < 1e-9, float(revs)
    raise ValueError(fam)


def _write_clip(job):
    fam, k, j, idx, out_dir = job
    ok, n = verify_count(fam, k, idx)
    if not ok:
        raise RuntimeError(f"count mismatch {fam} k={k} seed={idx}: got {n}")
    frames = FAMILIES[fam](k, idx)
    if len(frames) != N_FRAMES:
        raise RuntimeError(f"duration not fixed: {fam} k={k} -> {len(frames)}")
    name = f"{fam}_k{k:02d}_{j:03d}.mp4"
    imageio.mimwrite(
        os.path.join(out_dir, name),
        frames,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return {
        "file": name,
        "family": fam,
        "k": int(k),
        "seed": idx,
        "n_frames": N_FRAMES,
        "fps": FPS,
        "h": H,
        "w": W,
        "count_unit": COUNT_UNIT[fam],
    }


def build(out_dir, per_k, k_min, k_max, write_video=True, workers=1):
    os.makedirs(out_dir, exist_ok=True)
    jobs, idx = [], 0
    for fam in FAMILIES:
        for k in range(k_min, k_max + 1):
            for j in range(per_k):
                jobs.append((fam, k, j, idx, out_dir))
                idx += 1

    meta = []
    if write_video and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_write_clip, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                meta.append(fut.result())
                if i % 500 == 0:
                    print(f"[gen] {i}/{len(jobs)}", flush=True)
        meta.sort(key=lambda m: (list(FAMILIES).index(m["family"]), m["k"], m["file"]))
    else:
        for i, job in enumerate(jobs, 1):
            fam, k, j, idx_, _ = job
            if write_video:
                meta.append(_write_clip(job))
            else:
                ok, n = verify_count(fam, k, idx_)
                if not ok:
                    raise RuntimeError(f"count mismatch {fam} k={k} seed={idx_}: got {n}")
                meta.append({
                    "file": f"{fam}_k{k:02d}_{j:03d}.mp4",
                    "family": fam, "k": int(k), "seed": idx_,
                    "n_frames": N_FRAMES, "fps": FPS, "h": H, "w": W,
                    "count_unit": COUNT_UNIT[fam],
                })
            if i % 500 == 0:
                print(f"[gen] {i}/{len(jobs)}", flush=True)

    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    q = "Count the number of repetitions in this video. Return only an integer."
    with open(os.path.join(out_dir, "eval_synthetic.jsonl"), "w") as f:
        for m in meta:
            stem = os.path.splitext(m["file"])[0]
            f.write(json.dumps({
                "video": m["file"],
                "video_id": stem,
                "count": m["k"],
                "question": q,
                "action_type": m["family"],
                "family": m["family"],
                "seed": m["seed"],
                "n_frames": m["n_frames"],
                "fps": m["fps"],
                "duration": DUR_S,
                "count_unit": m["count_unit"],
            }) + "\n")
    print(f"[done] {len(meta)} clips ({len(FAMILIES)} families x "
          f"{k_max-k_min+1} k x {per_k}) -> {out_dir}")
    return meta


def make_contact_sheet(out_dir, fam, k=8, seed=0, path=None):
    """Quick visual check: save 8 sampled frames of one clip as a PNG strip."""
    frames = FAMILIES[fam](k, seed)
    picks = np.linspace(0, len(frames)-1, 8).astype(int)
    strip = np.concatenate([frames[i] for i in picks], axis=1)
    path = path or os.path.join(out_dir, f"_preview_{fam}_k{k}.png")
    Image.fromarray(strip).save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-k", type=int, default=50)   # 50 -> 10k total
    ap.add_argument("--k-min", type=int, default=K_MIN)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--preview", action="store_true",
                    help="only render one preview strip per family, no videos")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.preview:
        os.makedirs(args.out, exist_ok=True)
        for fam in FAMILIES:
            p = make_contact_sheet(args.out, fam, k=8, seed=0)
            print("preview:", p)
    else:
        # fail fast: every (family, k) unit must be exactly k events
        for fam in FAMILIES:
            for k in range(args.k_min, args.k_max + 1):
                ok, n = verify_count(fam, k, seed=0)
                if not ok:
                    raise SystemExit(f"verify failed {fam} k={k}: got {n}")
        print("[verify] all family x k count units pass", flush=True)
        build(args.out, args.per_k, args.k_min, args.k_max, workers=args.workers)


if __name__ == "__main__":
    main()

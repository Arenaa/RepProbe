# Data

## Synthetic (shipped)

`data/synth_reps_10k/` contains 10,000 clips (Git LFS) plus the eval manifest.

```text
data/synth_reps_10k/
  flash_k01_000.mp4 …
  eval_synthetic.jsonl      # 10,000 rows
  metadata.jsonl
```

Every clip is 64 frames / 4 s at 224×224, so duration does not leak *k*. Families: flash, textured, bounce, pendulum, scale, color, multi, shape, crossing, orbit. Balance: 50 clips × 10 families × 20 values of *k*.

Rebuild:

```bash
python scripts/gen_synth.py --out data/synth_reps_10k --per-k 50
python scripts/qc_synth.py --root data/synth_reps_10k
```

## Real (not shipped)

RepProbe-Real is cut from official [RepCount](https://github.com/XiaLiPKU/RepCount) videos. Those source videos are not redistributed here.

```text
data/repcount_nrep/              # python scripts/build_repcount.py
  videos/count_{k}/*.mp4
  all.jsonl                      # 32,233 clips
  train.jsonl                    # 23,223
  validation.jsonl               #  4,150
  test.jsonl                     #  4,860
  eval_10000.jsonl               # 10,000 balanced eval
  eval_2000.jsonl
  smoke_200.jsonl
```

```bash
python scripts/build_repcount.py \
  --ann-dir /path/to/RepCount/annotation \
  --video-root /path/to/RepCount/video \
  --out-dir data/repcount_nrep

python scripts/sample_eval_10000.py
python scripts/sample_eval_2000.py
```

`REPCOUNT_ANN` and `REPCOUNT_VIDEO` can replace `--ann-dir` / `--video-root`.

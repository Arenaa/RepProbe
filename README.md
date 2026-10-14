# RepProbe

A diagnostic benchmark for repetition counting in video-language models.

Each clip contains exactly *k* complete cycles, *k* ∈ [1, 20]. The question is always:

> Count the number of repetitions in this video. Return only an integer.

Counting accuracy collapses at high *k* across video-language models, yet a linear probe on the last-layer residual still recovers the count: the failure is at generation, not in what the model encodes.

- Paper: [arXiv:2610.18844](https://arxiv.org/abs/2610.18844)
- Project page: [arenaa.github.io/RepProbe-webpage](https://arenaa.github.io/RepProbe-webpage/)

This repository ships **code** and **RepProbe-Synthetic** (10,000 clips). Real RepCount videos are **not** included; rebuild them locally from the official source if you need RepProbe-Real.

```text
src/repprobe/
  dataset/     # real + synthetic clip builders
  eval/        # VLM / specialist scoring
  probe/       # hidden-state and head readouts
scripts/       # CLI entry points
data/
  synth_reps_10k/   # 10,000 synthetic clips (this release)
  repcount_nrep/    # real clips, gitignored — build locally
```

---

## Datasets

| | **RepProbe-Real** | **RepProbe-Synthetic** |
|---|---|---|
| Clips | 32,233 full pool · 10,000 balanced eval | **10,000 (shipped)** |
| Sources | 1,038 RepCount videos | 10 families × 1,000 clips |
| Count `k` | 1–20 | 1–20 |
| Balance | 500 clips / `k` in the 10k eval | 500 / `k` (50 per family) |
| Position | start / middle / end (~1/3 each) | n/a (fixed 4 s) |
| Duration | grows with `k` | **fixed** 4.0 s / 64 frames |
| In this repo | builder only | clips + `eval_synthetic.jsonl` |

Synthetic families: flash, textured, bounce, pendulum, scale, color, multi, shape, crossing, orbit. Clip length carries no information about *k*.

---

## Setup

```bash
git clone https://github.com/Arenaa/RepProbe.git
cd RepProbe

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # optional: python -m repprobe.eval.nrep
```

GPU recommended for model eval and probing. Linear probes also need `baukit` (`pip install baukit`). Set `HF_HOME` if Hugging Face weights live somewhere other than `~/.cache/huggingface`.

Run everything from the repo root. `python scripts/...` puts `src/` on `PYTHONPATH` automatically.

---

## 1. Data

### Synthetic (shipped)

Clips and the eval manifest live under `data/synth_reps_10k/`. Rebuild from scratch if you want:

```bash
python scripts/gen_synth.py --out data/synth_reps_10k --per-k 50
python scripts/qc_synth.py --root data/synth_reps_10k
```

`--per-k 50` → 10 families × 20 values of *k* × 50 = 10,000 clips.

### Real (RepCount n-rep, not shipped)

Needs official RepCount videos plus the per-cycle `L1, L2, …` annotation CSVs.

```bash
python scripts/build_repcount.py \
  --ann-dir /path/to/RepCount/annotation \
  --video-root /path/to/RepCount/video \
  --out-dir data/repcount_nrep

python scripts/sample_eval_10000.py   # 500 clips per k=1..20
python scripts/sample_eval_2000.py    # nested 100 per k
```

`REPCOUNT_ANN` and `REPCOUNT_VIDEO` can replace `--ann-dir` / `--video-root`.

---

## 2. Evaluation

Zero-shot VLM count eval (resume-safe JSONL):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_nrep.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --records data/synth_reps_10k/eval_synthetic.jsonl \
  --data-root data/synth_reps_10k \
  --out data/synth_reps_10k/preds/eval_synth__Qwen_Qwen3-VL-8B-Instruct.jsonl \
  --resume
```

Same harness on real clips after you build them (`--data-root data/repcount_nrep`, `--records data/repcount_nrep/eval_10000.jsonl`). Cosmos defaults to 4 fps; most other open models default to 16 frames.

Specialists (RepNet / ESCounts) on a local real build:

```bash
COUNTVID=/path/to/countVid CUDA_VISIBLE_DEVICES=0 \
  python scripts/eval_specialists.py \
  --model repnet --records data/repcount_nrep/eval_10000.jsonl --resume
```

Score overall / few / mid / many:

```bash
python scripts/score_bands.py data/repcount_nrep/preds/eval10000__*.jsonl
python scripts/score_synth.py
```

Shard helpers: `scripts/split_jsonl.py`, `scripts/merge_jsonl.py`.

A response with no parseable integer is scored as 0, then clipped to `[0, 40]`. Clips that were not run or crashed are excluded.

```bash
python scripts/rescore_obo.py --fail-as-zero --pool both \
  --json data/repcount_nrep/preds/obo_fail_as_zero.json
```

---

## 3. Probing

Frozen last-prompt-token residuals, logistic probe trained on 50 leftover clips per *k*:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_hidden.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --data-root data/repcount_nrep \
  --train-jsonl data/repcount_nrep/train.jsonl \
  --eval-jsonl data/repcount_nrep/eval_10000.jsonl \
  --layers last
```

Integer-set readout at the LM head (no labels):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_number_prior.py \
  --model-id nvidia/Cosmos-Reason2-8B \
  --records data/repcount_nrep/eval_10000.jsonl
```

Compare generation vs head-argmax vs last-layer probe (CPU, uses existing JSONLs):

```bash
python scripts/score_readouts.py
python scripts/score_readouts_qwen.py
```

Synthetic within-set probe (encode with `scripts/probe_hidden.py --eval-data-root data/synth_reps_10k`, then 5-fold CV):

```bash
python scripts/score_synth_cv.py
```

---

## Citation

```bibtex
@misc{khoshnoodi2026repprobevideolanguagemodels,
  title         = {RepProbe: Can Video Language Models Count Repeated Actions? A Diagnostic Benchmark},
  author        = {Mahsa Khoshnoodi and Sarah Adel Bargal},
  year          = {2026},
  eprint        = {2610.18844},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.18844}
}
```

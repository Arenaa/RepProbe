"""Repo-root paths. Independent of where a script lives."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
REPCOUNT = DATA_ROOT / "repcount_nrep"
SYNTH = DATA_ROOT / "synth_reps_10k"

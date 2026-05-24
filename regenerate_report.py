"""
Regenerate publication outputs (tables, plots, RESULTS_REPORT.md) from the
already-saved results in results/progress.jsonl — WITHOUT re-running any
training. Used to refresh the report after editing src/report.py.

Run:  python regenerate_report.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.config import RESULTS_DIR
from src.load_data import load_deap
from src.preprocess import preprocess
from src.report import generate_all_outputs

progress = os.path.join(RESULTS_DIR, "progress.jsonl")
results = [json.loads(l) for l in open(progress, encoding="utf-8") if l.strip()]
print(f"[regen] Loaded {len(results)} model-run results from progress.jsonl")

# Re-derive split metadata (deterministic — same seed) without training.
X, y, subjects = load_deap()
metas = {}
for protocol in ("subject_dependent", "subject_independent"):
    splits = preprocess(X, y, subjects, protocol)
    metas[protocol] = splits["meta"]

generate_all_outputs(results, metas)
print("[regen] Report and plots regenerated.")

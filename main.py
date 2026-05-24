"""
main.py — Full publication-grade EEG Emotion Recognition pipeline.

Comparative study of four models — SVM, Random Forest, EEGNet, and a Hybrid
Quantum-Classical model — for EEG-based valence/arousal classification on the
DEAP dataset, under TWO evaluation protocols:

  • subject-dependent   — stratified random 70/10/20 split
  • subject-independent — 22/4/6 participants held out (no subject leakage)

CRASH-RESILIENT: after each (protocol, model, task) finishes, its metrics are
appended to results/progress.jsonl and its predictions saved to a .npz file.
On restart the pipeline SKIPS already-completed runs, so an interruption
(e.g. the machine sleeping) does not lose finished work.

Usage:
    python main.py                  # full run, resumes if interrupted
    EEG_QML_FRESH=1 python main.py   # ignore previous progress, start over
    EEG_QML_QUICK=1 python main.py   # quick smoke test (few epochs, subsampled)
"""
import os
import sys
import json
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RESULTS_DIR, MODELS_DIR, QUICK, FRESH
from src.load_data import load_deap
from src.preprocess import preprocess
from src.features import extract_features
from src.classical import (set_seed, train_svm, predict_svm,
                            train_rf, predict_rf,
                            train_eegnet, predict_eegnet)
from src.quantum_model import train_quantum, predict_quantum
from src.evaluate import (compute_metrics, save_confusion_matrix,
                          save_training_curve, save_combined_roc)
from src.report import generate_all_outputs

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

PROTOCOLS = ["subject_dependent", "subject_independent"]
MODELS    = ["SVM", "RandomForest", "EEGNet", "HybridQNN"]
TASKS     = ["valence", "arousal"]

PROGRESS_FILE = os.path.join(RESULTS_DIR, "progress.jsonl")


# ── Resumability ───────────────────────────────────────────────────────────────

def load_progress() -> dict:
    """Return {(protocol, model, task): result_dict} for already-completed runs."""
    if FRESH and os.path.exists(PROGRESS_FILE):
        open(PROGRESS_FILE, "w").close()
        print("[resume] FRESH mode — previous progress cleared.")
        return {}
    done = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[(r["protocol"], r["model"], r["task"])] = r
                except json.JSONDecodeError:
                    pass
    return done


def append_progress(result: dict) -> None:
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def pred_path(protocol, model, task) -> str:
    return os.path.join(RESULTS_DIR, f"pred_{protocol}_{model}_{task}.npz")


# ── One model x task ───────────────────────────────────────────────────────────

def run_model_on_task(model_name, splits, task, protocol, all_results) -> None:
    """Train one model on one task, evaluate, save all artefacts + progress."""
    y_train = splits[f"y_train_{task}"]
    y_val   = splits[f"y_val_{task}"]
    y_test  = splits[f"y_test_{task}"]

    print(f"\n  ---- {model_name} / {task} / {protocol} ----")

    if model_name == "SVM":
        clf, train_time = train_svm(splits["X_train_pca"], y_train)
        preds, scores, inf_time = predict_svm(clf, splits["X_test_pca"])

    elif model_name == "RandomForest":
        clf, train_time = train_rf(splits["X_train_pca"], y_train)
        preds, scores, inf_time = predict_rf(clf, splits["X_test_pca"])

    elif model_name == "EEGNet":
        model, history, train_time = train_eegnet(
            splits["X_train"], y_train, splits["X_val"], y_val, task)
        preds, scores, inf_time = predict_eegnet(model, splits["X_test"])
        save_training_curve(history, "EEGNet", task, protocol)

    elif model_name == "HybridQNN":
        model, history, train_time = train_quantum(
            splits["X_train_pca"], y_train, splits["X_val_pca"], y_val, task)
        preds, scores, inf_time = predict_quantum(model, splits["X_test_pca"])
        save_training_curve(history, "HybridQNN", task, protocol)

    else:
        raise ValueError(f"Unknown model '{model_name}'")

    metrics = compute_metrics(y_test, preds, scores, train_time, inf_time)
    save_confusion_matrix(y_test, preds, model_name, task, protocol)

    # Save prediction outputs (required deliverable + lets ROC be rebuilt on resume)
    np.savez(pred_path(protocol, model_name, task),
             y_true=y_test, y_pred=preds, y_score=scores)

    result = {"protocol": protocol, "model": model_name, "task": task, **metrics}
    all_results.append(result)
    append_progress(result)                      # <-- crash-resilient checkpoint

    print(f"    Acc {metrics['accuracy']}%  F1 {metrics['f1']}%  "
          f"AUC {metrics['roc_auc']}  train {metrics['train_time_s']}s  [saved]")


def build_combined_roc(protocol: str) -> None:
    """Rebuild combined ROC curves from saved prediction files (resume-safe)."""
    for task in TASKS:
        roc_data = {}
        for model_name in MODELS:
            p = pred_path(protocol, model_name, task)
            if os.path.exists(p):
                d = np.load(p)
                roc_data[model_name] = (d["y_true"], d["y_score"])
        if roc_data:
            save_combined_roc(roc_data, task, protocol)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 64)
    print("  EEG EMOTION RECOGNITION — HYBRID QUANTUM-CLASSICAL STUDY")
    print("  Dual protocol: subject-dependent + subject-independent")
    if QUICK:
        print("  *** QUICK-TEST MODE — reduced epochs & subsampled data ***")
    print("=" * 64)

    set_seed()

    done = load_progress()
    if done:
        print(f"[resume] {len(done)} model-run(s) already complete — will skip them.")
    all_results = list(done.values())

    print("\n[Load] Reading dataset …")
    X_raw, y_raw, subjects = load_deap()

    metas = {}
    for protocol in PROTOCOLS:
        print("\n" + "#" * 64)
        print(f"#  PROTOCOL: {protocol.upper()}")
        print("#" * 64)

        print(f"\n[Preprocess] {protocol} …")
        splits = preprocess(X_raw, y_raw, subjects, protocol)

        # Feature extraction is only needed if at least one model still has work.
        pending = [(m, t) for m in MODELS for t in TASKS
                   if (protocol, m, t) not in done]
        if pending:
            print(f"\n[Features] {protocol} …")
            splits = extract_features(splits)
        else:
            print(f"[{protocol}] all models already complete — skipping features.")
        metas[protocol] = splits["meta"]

        for model_name in MODELS:
            for task in TASKS:
                if (protocol, model_name, task) in done:
                    print(f"  [skip] {model_name}/{task}/{protocol} — already done")
                    continue
                run_model_on_task(model_name, splits, task, protocol, all_results)

        build_combined_roc(protocol)
        print(f"\n[{protocol}] complete.")

    print("\n" + "#" * 64)
    print("#  GENERATING PUBLICATION OUTPUTS")
    print("#" * 64)
    generate_all_outputs(all_results, metas)

    print("\nDone. All metrics, plots and the report are in the results/ folder.")


if __name__ == "__main__":
    main()

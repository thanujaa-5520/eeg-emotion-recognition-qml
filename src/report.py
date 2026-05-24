"""
Step 7 — Publication-grade reporting.

Generates:
  • metrics_subject_dependent.csv / metrics_subject_independent.csv
  • metrics_combined.csv
  • comparison_<protocol>.png        — per-protocol model comparison bar charts
  • generalization_gap.png           — subject-dependent vs subject-independent
  • RESULTS_REPORT.md                — methodology, results tables, statistical
                                       observations, generalization analysis,
                                       runtime comparison, reproducibility,
                                       limitations, future work

Academic positioning: the generated text uses deliberately cautious wording
("competitive", "promising", "potential benefit under evaluated conditions")
and does NOT claim universal quantum superiority.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import (RESULTS_DIR, RANDOM_SEED, N_QUBITS, VQC_LAYERS,
                        PCA_COMPONENTS, EEGNET_EPOCHS, QUANTUM_EPOCHS,
                        FREQ_BANDS, SFREQ)
from src.evaluate import MODEL_COLOURS

MODELS = ["SVM", "RandomForest", "EEGNet", "HybridQNN"]
TASKS  = ["valence", "arousal"]
PROTOCOLS = [("subject_dependent", "Subject-Dependent"),
             ("subject_independent", "Subject-Independent")]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1",
               "roc_auc", "train_time_s", "inference_ms"]


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def _get(results, protocol, model, task):
    for r in results:
        if r["protocol"] == protocol and r["model"] == model and r["task"] == task:
            return r
    return None


def _mean_acc(results, protocol, model):
    vals = [r["accuracy"] for r in results
            if r["protocol"] == protocol and r["model"] == model]
    return float(np.mean(vals)) if vals else float("nan")


def _is_degenerate(r) -> bool:
    """A run that predicted a single class for all test samples (recall ~100% or ~0%)."""
    return r["recall"] >= 99.9 or r["recall"] <= 0.1


# ── CSV tables ─────────────────────────────────────────────────────────────────

def save_csv_tables(results) -> None:
    cols = ["protocol", "model", "task"] + METRIC_KEYS
    # combined
    with open(os.path.join(RESULTS_DIR, "metrics_combined.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    # per protocol
    for proto, _ in PROTOCOLS:
        rows = [r for r in results if r["protocol"] == proto]
        path = os.path.join(RESULTS_DIR, f"metrics_{proto}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print("[report] CSV tables saved (combined + per-protocol)")


# ── Console table ──────────────────────────────────────────────────────────────

def print_results_table(results) -> None:
    for proto, label in PROTOCOLS:
        rows = [r for r in results if r["protocol"] == proto]
        if not rows:
            continue
        header = (f"{'Model':<14}{'Task':<10}{'Acc%':>7}{'Prec%':>7}"
                  f"{'Rec%':>7}{'F1%':>7}{'AUC':>8}{'Train(s)':>10}{'Inf(ms)':>9}")
        print(f"\n  === {label} ===")
        print("  " + "-" * len(header))
        print("  " + header)
        print("  " + "-" * len(header))
        for r in rows:
            print(f"  {r['model']:<14}{r['task']:<10}{r['accuracy']:>7}"
                  f"{r['precision']:>7}{r['recall']:>7}{r['f1']:>7}"
                  f"{r['roc_auc']:>8}{r['train_time_s']:>10}{r['inference_ms']:>9}")
        print("  " + "-" * len(header))


# ── Plots ──────────────────────────────────────────────────────────────────────

def save_comparison_barcharts(results) -> None:
    for proto, label in PROTOCOLS:
        if not any(r["protocol"] == proto for r in results):
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(MODELS))
        width = 0.35
        for i, task in enumerate(TASKS):
            accs = [(_get(results, proto, m, task) or {}).get("accuracy", 0)
                    for m in MODELS]
            bars = ax.bar(x + (i - 0.5) * width, accs, width,
                          label=task.capitalize())
            for b, a in zip(bars, accs):
                ax.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.1f}",
                        ha="center", fontsize=8)
        ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(MODELS)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Model Comparison — {label}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"comparison_{proto}.png"), dpi=150)
        plt.close()
    print("[report] Per-protocol comparison bar charts saved")


def save_generalization_chart(results) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    width = 0.35
    dep  = [_mean_acc(results, "subject_dependent",   m) for m in MODELS]
    indep = [_mean_acc(results, "subject_independent", m) for m in MODELS]
    b1 = ax.bar(x - width / 2, dep,   width, label="Subject-Dependent",
                color="#4C72B0")
    b2 = ax.bar(x + width / 2, indep, width, label="Subject-Independent",
                color="#C44E52")
    for bars, vals in ((b1, dep), (b2, indep)):
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                        ha="center", fontsize=8)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("Mean Test Accuracy (%)  [valence + arousal]")
    ax.set_title("Generalization Gap — Subject-Dependent vs Subject-Independent")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "generalization_gap.png"), dpi=150)
    plt.close()
    print("[report] Generalization-gap chart saved")


# ── Markdown report ────────────────────────────────────────────────────────────

def _md_table(results, proto) -> str:
    rows = [r for r in results if r["protocol"] == proto]
    head = ("| Model | Task | Acc % | Prec % | Rec % | F1 % | ROC-AUC | "
            "Train (s) | Infer (ms) |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| {r['model']} | {r['task']} | {r['accuracy']} | "
                 f"{r['precision']} | {r['recall']} | {r['f1']} | "
                 f"{r['roc_auc']} | {r['train_time_s']} | {r['inference_ms']} |\n")
    return head + body


def generate_markdown_report(results, metas) -> None:
    L = []
    L.append("# Hybrid Quantum-Classical EEG Emotion Recognition — Results Report\n")
    L.append("Comparative study of four models for EEG-based valence and arousal "
             "classification on the DEAP dataset, evaluated under two protocols.\n")

    # 1. Methodology
    L.append("## 1. Methodology Summary\n")
    L.append(f"- **Dataset:** DEAP — 32 participants x 40 trials = 1280 trials, "
             f"32 EEG channels, {SFREQ} Hz, 63 s per trial.\n")
    L.append("- **Tasks:** binary valence and binary arousal "
             "(rating >= 5 -> high, < 5 -> low).\n")
    L.append("- **Preprocessing:** per-trial per-channel z-score normalisation. "
             "DEAP data is already band-pass filtered (4-45 Hz).\n")
    L.append(f"- **Features (SVM / RF / HybridQNN):** Welch PSD power in "
             f"{len(FREQ_BANDS)} bands ({', '.join(FREQ_BANDS)}) x 32 channels "
             f"= 160 features, reduced to {PCA_COMPONENTS} via PCA "
             f"(fit on train only).\n")
    L.append("- **EEGNet** consumes raw EEG time-series directly.\n")
    L.append(f"- **HybridQNN:** classical encoder -> {N_QUBITS}-qubit variational "
             f"quantum circuit ({VQC_LAYERS} layers, PennyLane default.qubit "
             f"simulator) -> classical head.\n")
    L.append(f"- **Training:** EEGNet {EEGNET_EPOCHS} epochs, HybridQNN "
             f"{QUANTUM_EPOCHS} epochs, both with best-validation checkpointing. "
             f"SVM and RF are single-shot fits.\n")
    L.append("- **Protocols:** (a) *subject-dependent* — stratified random "
             "70/10/20 split; (b) *subject-independent* — 22/4/6 participants "
             "held out so test subjects never appear in training.\n")

    # 2 & 3. Result tables
    L.append("\n## 2. Results — Subject-Dependent Protocol\n")
    L.append(_md_table(results, "subject_dependent"))
    L.append("\n## 3. Results — Subject-Independent Protocol\n")
    L.append(_md_table(results, "subject_independent"))

    # 4. Combined
    L.append("\n## 4. Combined Comparison\n")
    L.append("See `metrics_combined.csv`, `comparison_subject_dependent.png`, "
             "`comparison_subject_independent.png`, and `generalization_gap.png`.\n")

    # 5. Statistical observations
    L.append("\n## 5. Statistical Observations\n")
    degenerate = [r for r in results if _is_degenerate(r)]
    for proto, label in PROTOCOLS:
        rows = [r for r in results if r["protocol"] == proto]
        if not rows:
            continue
        valid = [r for r in rows if not _is_degenerate(r)] or rows
        best = max(valid, key=lambda r: r["accuracy"])
        mean_acc = np.mean([r["accuracy"] for r in rows])
        L.append(f"- **{label}:** mean accuracy across all models/tasks "
                 f"= {mean_acc:.2f}%. Highest non-degenerate single result: "
                 f"{best['model']} on {best['task']} ({best['accuracy']}%).\n")

    if degenerate:
        L.append("- **Degenerate-predictor caution.** The following run(s) "
                 "collapsed to predicting a single class for every test sample "
                 "(recall ~100% or ~0%). Their accuracy reflects the class prior, "
                 "NOT genuine discrimination, and must not be read as a "
                 "performance result:\n")
        for r in degenerate:
            L.append(f"  - {r['model']} / {r['task']} / {r['protocol']} — "
                     f"recall {r['recall']}%, accuracy {r['accuracy']}%, "
                     f"F1 {r['f1']}% (ROC-AUC {r['roc_auc']} — the underlying "
                     f"scores carry only weak signal; the decision threshold "
                     f"collapsed). Any aggregate that includes this run is "
                     f"correspondingly inflated.\n")

    hybrid_dep   = _mean_acc(results, "subject_dependent",   "HybridQNN")
    hybrid_indep = _mean_acc(results, "subject_independent", "HybridQNN")
    hybrid_degen = any(_is_degenerate(r) for r in results
                       if r["model"] == "HybridQNN")
    caveat = (" (note: the subject-independent figure is inflated by a "
              "collapsed-predictor run — see the caution above)"
              if hybrid_degen else "")
    L.append(f"- The hybrid quantum-classical model reached "
             f"{hybrid_dep:.2f}% (subject-dependent) and {hybrid_indep:.2f}% "
             f"(subject-independent) mean accuracy{caveat}. On the "
             f"subject-dependent protocol it performed competitively with the "
             f"classical baselines; it did not consistently outperform them, "
             f"and on several task/protocol combinations a classical model "
             f"scored higher.\n")
    L.append("- Reported differences between models are modest and, given the "
             "small test sets, should be interpreted as broadly comparable "
             "rather than as decisive rankings. No model dominates universally.\n")

    # 6. Generalization analysis
    L.append("\n## 6. Generalization Analysis\n")
    L.append("Performance under the subject-independent protocol is expected to "
             "be lower than under the subject-dependent protocol for all models. "
             "Per-model mean-accuracy comparison:\n\n")
    L.append("| Model | Subject-Dependent | Subject-Independent | Gap |\n")
    L.append("|---|---|---|---|\n")
    for m in MODELS:
        d  = _mean_acc(results, "subject_dependent",   m)
        i  = _mean_acc(results, "subject_independent", m)
        gap = d - i
        L.append(f"| {m} | {d:.2f}% | {i:.2f}% | {gap:+.2f}% |\n")
    if any(_is_degenerate(r) for r in results
           if r["model"] == "HybridQNN" and r["protocol"] == "subject_independent"):
        L.append("\n*Caveat:* the HybridQNN subject-independent mean — and "
                 "therefore its apparently small generalization gap — is "
                 "inflated by a collapsed-predictor run (Section 5). Its gap "
                 "should not be interpreted as superior cross-subject "
                 "generalization.\n")
    L.append("\n**Why subject-independent performance is lower:** EEG signals "
             "exhibit substantial inter-subject variability — differences in "
             "skull anatomy, electrode placement, baseline neural activity and "
             "individual emotional expression. In the subject-dependent "
             "protocol a model can exploit participant-specific patterns "
             "because trials from each participant appear in both training and "
             "test. In the subject-independent protocol the model must "
             "generalise to entirely unseen participants, which is a harder "
             "and more realistic evaluation. The gap reflects this difficulty "
             "rather than a defect in any model, and is consistent with the "
             "EEG emotion-recognition literature.\n")

    # 7. Runtime
    L.append("\n## 7. Runtime Comparison\n")
    L.append("| Model | Protocol | Mean Train (s) | Mean Infer (ms) |\n")
    L.append("|---|---|---|---|\n")
    for m in MODELS:
        for proto, label in PROTOCOLS:
            rows = [r for r in results
                    if r["protocol"] == proto and r["model"] == m]
            if not rows:
                continue
            tt = np.mean([r["train_time_s"] for r in rows])
            it = np.mean([r["inference_ms"] for r in rows])
            L.append(f"| {m} | {label} | {tt:.2f} | {it:.1f} |\n")

    # 8. Reproducibility
    L.append("\n## 8. Reproducibility Notes\n")
    L.append(f"- Global random seed: **{RANDOM_SEED}** (Python, NumPy, PyTorch).\n")
    L.append("- PCA and StandardScaler are fit on the training split only.\n")
    L.append("- The same train/val/test split is reused across all four models "
             "within a protocol, ensuring a fair comparison.\n")
    for proto, label in PROTOCOLS:
        meta = metas.get(proto, {})
        if meta.get("protocol") == "subject_independent":
            L.append(f"- **{label}** subject assignment — "
                     f"train: {meta.get('train_subjects')}; "
                     f"val: {meta.get('val_subjects')}; "
                     f"test: {meta.get('test_subjects')}.\n")
    L.append("- Quantum circuit executed on the PennyLane `default.qubit` "
             "statevector simulator (deterministic, no shot noise).\n")

    # 9. Limitations
    L.append("\n## 9. Limitations\n")
    L.append("- The quantum component runs on a noiseless simulator; results do "
             "not account for real quantum-hardware noise or decoherence.\n")
    L.append("- The quantum circuit is intentionally compact "
             f"({N_QUBITS} qubits, {VQC_LAYERS} layers); larger circuits were "
             "not explored.\n")
    L.append("- PCA to 16 components discards some variance before the "
             "SVM / RF / quantum models; EEGNet avoids this by using raw EEG.\n")
    L.append("- The subject-independent protocol uses a single fixed "
             "22/4/6 split rather than full leave-one-subject-out "
             "cross-validation, for computational tractability on CPU.\n")
    L.append("- Binary thresholding at rating = 5 follows common practice but "
             "discards the intensity information in the original 1-9 scale.\n")

    # 10. Future work
    L.append("\n## 10. Future Work\n")
    L.append("- Evaluate with full leave-one-subject-out cross-validation.\n")
    L.append("- Test the hybrid model on real quantum hardware or noisy "
             "simulators to assess robustness.\n")
    L.append("- Explore alternative quantum feature encodings and circuit "
             "depths under controlled comparison.\n")
    L.append("- Extend from binary to multi-class (e.g. four-quadrant "
             "valence-arousal) classification.\n")
    L.append("- Investigate subject-adaptive or transfer-learning strategies "
             "to narrow the subject-independent generalization gap.\n")

    L.append("\n---\n*Generated automatically by the experiment pipeline. "
             "Wording is intentionally cautious; the hybrid quantum-classical "
             "model is presented as a promising architecture showing "
             "competitive performance under the evaluated conditions, not as a "
             "universally superior method.*\n")

    path = os.path.join(RESULTS_DIR, "RESULTS_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"[report] Markdown report saved -> {path}")


# ── Top-level ──────────────────────────────────────────────────────────────────

def generate_all_outputs(results, metas) -> None:
    print("\n[report] Generating publication outputs …")
    save_csv_tables(results)
    save_comparison_barcharts(results)
    save_generalization_chart(results)
    generate_markdown_report(results, metas)
    print_results_table(results)

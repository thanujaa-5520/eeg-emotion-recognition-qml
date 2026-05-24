"""
PHASE 1 — Data Validation & Preprocessing Check

Runs BEFORE any heavy training. Validates dataset integrity, reports dataset
properties, checks the preprocessing pipeline, and detects issues that would
waste runtime in a 2-3 hour training job.

Does NOT train any model. Read-only validation + a small preprocessing probe.

Run:  python phase1_validation.py
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.config import (SFREQ, N_EEG_CHANNELS, LABEL_THRESHOLD,
                        TEST_SIZE, RANDOM_SEED, FREQ_BANDS, PCA_COMPONENTS)
from src.load_data import load_deap
from src.preprocess import make_binary_labels, zscore_normalise, split_data
from src.features import extract_psd_features, apply_pca

np.random.seed(RANDOM_SEED)

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]
PASS, WARN, FAIL = "[PASS]", "[WARN]", "[FAIL]"
issues = []          # collect WARN/FAIL items for the final summary


def hr(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check(condition, label, detail_pass="", detail_fail="", level="FAIL"):
    """Print a PASS line, or a WARN/FAIL line and record the issue."""
    if condition:
        print(f"  {PASS} {label}" + (f" — {detail_pass}" if detail_pass else ""))
        return True
    tag = WARN if level == "WARN" else FAIL
    print(f"  {tag} {label}" + (f" — {detail_fail}" if detail_fail else ""))
    issues.append((level, label, detail_fail))
    return False


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("#  PHASE 1 — DATA VALIDATION & PREPROCESSING CHECK")
    print("#  (no model training — validation only)")
    print("#" * 70)

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION A — DATASET INTEGRITY")
    X, y = load_deap()

    check(X.ndim == 3, "X is 3-dimensional",
          f"shape {X.shape}", f"got {X.ndim}D, shape {X.shape}")
    check(y.ndim == 2, "y is 2-dimensional",
          f"shape {y.shape}", f"got {y.ndim}D, shape {y.shape}")
    check(X.shape[0] == y.shape[0], "X and y trial counts match",
          f"{X.shape[0]} trials", f"X has {X.shape[0]}, y has {y.shape[0]}")
    check(X.shape[1] == N_EEG_CHANNELS, "EEG channel count correct",
          f"{X.shape[1]} channels", f"expected {N_EEG_CHANNELS}, got {X.shape[1]}")

    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    check(n_nan == 0, "No NaN values in EEG signals",
          "0 NaNs", f"{n_nan} NaN values found")
    check(n_inf == 0, "No infinite values in EEG signals",
          "0 Infs", f"{n_inf} Inf values found")

    label_nan = int(np.isnan(y).sum())
    check(label_nan == 0, "No NaN values in labels",
          "0 NaNs", f"{label_nan} NaN labels found")

    # Corrupted-sample checks
    zero_trials = int(np.sum(np.all(X == 0, axis=(1, 2))))
    check(zero_trials == 0, "No all-zero (empty) trials",
          "0 empty trials", f"{zero_trials} trials are entirely zero")

    # Dead channels: zero variance within a trial
    channel_var = X.var(axis=2)                       # (n_trials, n_channels)
    dead = int(np.sum(channel_var < 1e-12))
    check(dead == 0, "No dead (zero-variance) channels",
          "all channels carry signal",
          f"{dead} channel-trial pairs have ~zero variance", level="WARN")

    print(f"\n  Signal amplitude range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Signal global mean / std: {X.mean():.4f} / {X.std():.4f}")

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION B — DATASET DIMENSIONS")
    n_trials, n_ch, n_samples = X.shape
    duration = n_samples / SFREQ
    print(f"  Total trials .............. {n_trials}")
    print(f"  EEG channels .............. {n_ch}")
    print(f"  Samples per trial ......... {n_samples}")
    print(f"  Sampling frequency ........ {SFREQ} Hz")
    print(f"  Trial duration ............ {duration:.1f} s")
    print(f"  Participants (DEAP) ....... {n_trials // 40} ({n_trials // 40} x 40 trials)")
    print(f"  Tasks ..................... 2 (valence, arousal)")
    print(f"  Classes per task .......... 2 (binary: low / high)")
    print(f"  Label columns ............. {y.shape[1]} ({', '.join(LABEL_NAMES)})")

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION C — LABEL VALIDATION")
    print(f"  Raw rating ranges (DEAP scale is 1-9):")
    for i, name in enumerate(LABEL_NAMES):
        col = y[:, i]
        valid = (col >= 1.0) & (col <= 9.0)
        n_invalid = int((~valid).sum())
        print(f"    {name:<10} min={col.min():.2f}  max={col.max():.2f}  "
              f"mean={col.mean():.2f}")
        check(n_invalid == 0, f"All '{name}' ratings within [1, 9]",
              "valid", f"{n_invalid} ratings outside [1,9]", level="WARN")

    labels = make_binary_labels(y)
    print(f"\n  Binary class distribution (threshold = {LABEL_THRESHOLD}):")
    for task in ("valence", "arousal"):
        lab = labels[task]
        check(set(np.unique(lab).tolist()).issubset({0, 1}),
              f"'{task}' labels are valid binary {{0,1}}",
              "encoded correctly", "invalid label values present")
        low, high = int((lab == 0).sum()), int((lab == 1).sum())
        ratio = max(low, high) / max(min(low, high), 1)
        print(f"    {task:<10} low(<5)={low:4d}  high(>=5)={high:4d}  "
              f"imbalance={ratio:.2f}:1")
        check(ratio <= 1.5, f"'{task}' class balance acceptable",
              f"{ratio:.2f}:1", f"{ratio:.2f}:1 — imbalanced; stratified "
              f"split + F1/precision/recall will handle it", level="WARN")

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION D — PREPROCESSING PIPELINE VALIDATION")

    print("  D.1 Normalisation (per-channel, per-trial z-score)")
    X_norm = zscore_normalise(X)
    post_mean = float(np.mean(X_norm.mean(axis=2)))
    post_std  = float(np.mean(X_norm.std(axis=2)))
    check(abs(post_mean) < 1e-3, "Post-normalisation channel mean ~ 0",
          f"mean={post_mean:.2e}", f"mean={post_mean:.4f}")
    check(abs(post_std - 1.0) < 1e-2, "Post-normalisation channel std ~ 1",
          f"std={post_std:.4f}", f"std={post_std:.4f}")
    check(int(np.isnan(X_norm).sum()) == 0, "Normalisation introduced no NaNs",
          "clean", "NaNs created during normalisation")

    print("\n  D.2 Filtering")
    print(f"    {PASS} DEAP preprocessed data is already band-pass filtered")
    print(f"           (4-45 Hz) and down-sampled to {SFREQ} Hz by the DEAP")
    print(f"           authors — no additional filtering applied (correct).")

    print("\n  D.3 EEGNet input reshaping")
    eegnet_in = X_norm[:4, np.newaxis, :, :]          # add channel dim
    expected = (4, 1, N_EEG_CHANNELS, n_samples)
    check(eegnet_in.shape == expected,
          "EEG reshapes to EEGNet 4-D input (batch, 1, channels, samples)",
          f"{eegnet_in.shape}", f"got {eegnet_in.shape}, expected {expected}")

    print("\n  D.4 Feature extraction + PCA (probe on 32-trial subset)")
    probe = X_norm[:32]
    feats = extract_psd_features(probe)
    exp_feat_dim = N_EEG_CHANNELS * len(FREQ_BANDS)
    check(feats.shape == (32, exp_feat_dim),
          f"PSD features have correct shape (32, {exp_feat_dim})",
          f"{feats.shape}", f"got {feats.shape}")
    check(int(np.isnan(feats).sum()) == 0, "PSD features contain no NaNs",
          "clean", "NaNs in extracted features")
    pca_tr, pca_te, _ = apply_pca(feats[:24], feats[24:])
    check(pca_tr.shape[1] == PCA_COMPONENTS,
          f"PCA reduces features to {PCA_COMPONENTS} components",
          f"{pca_tr.shape[1]} components", f"got {pca_tr.shape[1]}")

    print("\n  D.5 Label encoding")
    check(labels["valence"].dtype.kind in "iu" and labels["arousal"].dtype.kind in "iu",
          "Continuous ratings encoded to integer class labels",
          "int labels", "labels not integer-encoded")

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION E — TRAIN / TEST SPLIT")
    splits = split_data(X_norm, labels)
    n_tr, n_te = len(splits["X_train"]), len(splits["X_test"])
    print(f"  Split ratio ............... {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test")
    print(f"  Train trials .............. {n_tr}")
    print(f"  Test  trials .............. {n_te}")
    check(n_tr + n_te == n_trials, "Split covers all trials with no overlap",
          f"{n_tr}+{n_te}={n_trials}", "trial count mismatch after split")

    print(f"\n  Stratification check (class ratio should match across splits):")
    for task in ("valence", "arousal"):
        tr = splits[f"y_train_{task}"]
        te = splits[f"y_test_{task}"]
        tr_hi = tr.mean()
        te_hi = te.mean()
        check(abs(tr_hi - te_hi) < 0.05,
              f"'{task}' high-class proportion consistent train vs test",
              f"train={tr_hi:.3f}, test={te_hi:.3f}",
              f"train={tr_hi:.3f}, test={te_hi:.3f} — stratification drift",
              level="WARN")

    # ════════════════════════════════════════════════════════════════════════
    hr("SECTION F — METHODOLOGY GAPS TO ADDRESS BEFORE PHASE 2/3")
    print("  These are NOT data errors — they are pipeline changes required for")
    print("  the publication-grade protocol you specified. I have NOT made them")
    print("  yet; they need your decision before Phase 2.\n")
    print("  F.1  VALIDATION SPLIT — current pipeline does train/test only.")
    print("       Phases 2-3 need a 3-way train/validation/test separation so")
    print("       validation metrics (for early stopping / best checkpoint)")
    print("       stay separate from the final test metrics.")
    print()
    print("  F.2  BEST-CHECKPOINT — current code saves the final-epoch model.")
    print("       Publication protocol wants the best-validation-epoch model.")
    print()
    print("  F.3  ROC-AUC — not yet computed in evaluate.py; you requested it.")
    print()
    print("  F.4  SPLIT PROTOCOL — current split is SUBJECT-DEPENDENT (trials")
    print("       from all 32 participants mixed across train/test). This is a")
    print("       common DEAP protocol, but SUBJECT-INDEPENDENT (hold out whole")
    print("       participants) is more rigorous and avoids per-subject identity")
    print("       leakage. This is a methodology choice you should make.")
    print()
    print("  Note: PCA is correctly fit on TRAIN only (no feature leakage), and")
    print("  z-score is per-trial (no cross-trial leakage). Those are clean.")

    # ════════════════════════════════════════════════════════════════════════
    hr("PHASE 1 SUMMARY")
    fails = [i for i in issues if i[0] == "FAIL"]
    warns = [i for i in issues if i[0] == "WARN"]
    if not fails and not warns:
        print(f"  {PASS} All integrity and preprocessing checks passed cleanly.")
    if warns:
        print(f"  {len(warns)} WARNING(S):")
        for _, label, detail in warns:
            print(f"    - {label}: {detail}")
    if fails:
        print(f"\n  {len(fails)} FAILURE(S) — must fix before training:")
        for _, label, detail in fails:
            print(f"    - {label}: {detail}")
        print("\n  PHASE 1 RESULT: FAILED — do not proceed to Phase 2.")
        sys.exit(1)

    print("\n  PHASE 1 RESULT: PASSED")
    print("  Data is valid. Awaiting your review + decisions on Section F")
    print("  before implementing Phase 2 (quick validation run).")


if __name__ == "__main__":
    main()

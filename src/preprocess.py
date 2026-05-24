"""
Step 2 — Preprocessing & splitting (two protocols).

- Z-score normalise each channel independently per trial
- Create binary labels (valence and arousal) using threshold >= 5
- Build a 3-way train / validation / test split under one of two protocols:

  SUBJECT-DEPENDENT   — stratified random split across all trials.
                        Trials from the same participant may appear in both
                        train and test. 70 / 10 / 20.

  SUBJECT-INDEPENDENT — split by participant. Whole subjects are assigned to
                        exactly one of train / val / test, so test subjects
                        are NEVER seen during training. 22 / 4 / 6 subjects.

Both protocols return an identical dict structure so downstream code is shared.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (LABEL_THRESHOLD, VALENCE_COL, AROUSAL_COL,
                        RANDOM_SEED, QUICK)

# ── Split configuration ────────────────────────────────────────────────────────
SD_TEST_FRAC = 0.20          # subject-dependent: test fraction
SD_VAL_FRAC  = 0.10          # subject-dependent: validation fraction (of total)
SD_QUICK_TRIALS = 240        # subject-dependent quick-mode subsample

SI_TRAIN_SUBJ = 22           # subject-independent: train participants
SI_VAL_SUBJ   = 4            # subject-independent: validation participants
SI_TEST_SUBJ  = 6            # subject-independent: test participants
SI_QUICK_SUBJ = (6, 2, 2)    # subject-independent quick-mode (train, val, test)

TASKS = ("valence", "arousal")


# ── Label / normalisation helpers ──────────────────────────────────────────────

def make_binary_labels(y_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Convert continuous DEAP ratings (1-9) to binary class labels."""
    return {
        "valence": (y_raw[:, VALENCE_COL] >= LABEL_THRESHOLD).astype(int),
        "arousal": (y_raw[:, AROUSAL_COL] >= LABEL_THRESHOLD).astype(int),
    }


def zscore_normalise(X: np.ndarray) -> np.ndarray:
    """Z-score per channel per trial. X shape: (n, channels, samples)."""
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


# ── Split dict builder ─────────────────────────────────────────────────────────

def _build_split_dict(X, labels, subjects,
                      idx_train, idx_val, idx_test, meta) -> dict:
    d = {
        "X_train": X[idx_train], "X_val": X[idx_val], "X_test": X[idx_test],
        "subjects_train": subjects[idx_train],
        "subjects_val":   subjects[idx_val],
        "subjects_test":  subjects[idx_test],
        "meta": meta,
    }
    for task in TASKS:
        d[f"y_train_{task}"] = labels[task][idx_train]
        d[f"y_val_{task}"]   = labels[task][idx_val]
        d[f"y_test_{task}"]  = labels[task][idx_test]
    return d


# ── Protocol 1: subject-dependent ──────────────────────────────────────────────

def split_subject_dependent(X, labels, subjects) -> dict:
    """Stratified random 70/10/20 split across all trials (stratified on valence)."""
    idx = np.arange(len(X))

    if QUICK and len(idx) > SD_QUICK_TRIALS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(idx, SD_QUICK_TRIALS, replace=False))

    strat = labels["valence"][idx]
    idx_trainval, idx_test = train_test_split(
        idx, test_size=SD_TEST_FRAC, stratify=strat, random_state=RANDOM_SEED)

    # validation fraction expressed relative to the train+val pool
    val_rel = SD_VAL_FRAC / (1.0 - SD_TEST_FRAC)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_rel,
        stratify=labels["valence"][idx_trainval], random_state=RANDOM_SEED)

    meta = {"protocol": "subject_dependent",
            "note": "stratified random split; participants may span splits"}
    return _build_split_dict(X, labels, subjects,
                             idx_train, idx_val, idx_test, meta)


# ── Protocol 2: subject-independent ────────────────────────────────────────────

def split_subject_independent(X, labels, subjects) -> dict:
    """
    Cross-subject split. Whole participants are assigned to exactly one of
    train / val / test — no subject appears in more than one split.
    """
    unique_subj = np.unique(subjects)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(unique_subj)

    if QUICK:
        n_tr, n_va, n_te = SI_QUICK_SUBJ
    else:
        n_tr, n_va, n_te = SI_TRAIN_SUBJ, SI_VAL_SUBJ, SI_TEST_SUBJ

    needed = n_tr + n_va + n_te
    if len(shuffled) < needed:
        raise ValueError(f"[preprocess] Need {needed} subjects for subject-"
                         f"independent split, only {len(shuffled)} available.")
    shuffled = shuffled[:needed]

    test_subj  = np.sort(shuffled[:n_te])
    val_subj   = np.sort(shuffled[n_te:n_te + n_va])
    train_subj = np.sort(shuffled[n_te + n_va:])

    # ── Hard leakage guard ──
    s_tr, s_va, s_te = set(train_subj), set(val_subj), set(test_subj)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise RuntimeError("[preprocess] SUBJECT LEAKAGE detected — aborting.")

    idx_train = np.where(np.isin(subjects, train_subj))[0]
    idx_val   = np.where(np.isin(subjects, val_subj))[0]
    idx_test  = np.where(np.isin(subjects, test_subj))[0]

    meta = {
        "protocol": "subject_independent",
        "train_subjects": train_subj.tolist(),
        "val_subjects":   val_subj.tolist(),
        "test_subjects":  test_subj.tolist(),
        "note": "whole participants held out — no subject leakage",
    }
    return _build_split_dict(X, labels, subjects,
                             idx_train, idx_val, idx_test, meta)


# ── Main preprocessing entry point ─────────────────────────────────────────────

def preprocess(X_raw: np.ndarray, y_raw: np.ndarray,
               subjects: np.ndarray, protocol: str) -> dict:
    """
    protocol : "subject_dependent" or "subject_independent"
    """
    print(f"[preprocess] Protocol = {protocol}")
    print("[preprocess] Normalising EEG channels (per-trial z-score) …")
    X_norm = zscore_normalise(X_raw)
    labels = make_binary_labels(y_raw)

    if protocol == "subject_dependent":
        splits = split_subject_dependent(X_norm, labels, subjects)
    elif protocol == "subject_independent":
        splits = split_subject_independent(X_norm, labels, subjects)
    else:
        raise ValueError(f"[preprocess] Unknown protocol '{protocol}'")

    _report_split(splits)
    return splits


def _report_split(splits: dict) -> None:
    n_tr = len(splits["X_train"])
    n_va = len(splits["X_val"])
    n_te = len(splits["X_test"])
    print(f"[preprocess] Trials — train: {n_tr}, val: {n_va}, test: {n_te}")

    meta = splits["meta"]
    if meta["protocol"] == "subject_independent":
        print(f"[preprocess]   train subjects: {meta['train_subjects']}")
        print(f"[preprocess]   val   subjects: {meta['val_subjects']}")
        print(f"[preprocess]   test  subjects: {meta['test_subjects']}")
        # explicit leakage verification on the realised index sets
        s_tr = set(splits["subjects_train"].tolist())
        s_te = set(splits["subjects_test"].tolist())
        s_va = set(splits["subjects_val"].tolist())
        leak = (s_tr & s_te) | (s_tr & s_va) | (s_va & s_te)
        print(f"[preprocess]   leakage check: "
              f"{'NONE — clean' if not leak else f'LEAK {leak}'}")

    for task in TASKS:
        tr = splits[f"y_train_{task}"].mean()
        va = splits[f"y_val_{task}"].mean()
        te = splits[f"y_test_{task}"].mean()
        print(f"[preprocess]   {task} high-class proportion — "
              f"train: {tr:.3f}, val: {va:.3f}, test: {te:.3f}")

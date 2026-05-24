"""
Step 1 — Load the EEG emotion dataset (format-agnostic).

Auto-detects whatever is placed in the data/ folder:
  • s01.dat … s32.dat   → original DEAP per-subject pickle files
  • *.dat (single)      → one combined / single-subject pickle
  • data.npy + labels.npy → NumPy array pair
  • *.npz               → single NumPy archive with data + label arrays
  • *.csv               → flattened CSV (common on Kaggle DEAP mirrors)
If the folder is empty, synthetic demo data is generated.

Returns
-------
X        : (n_trials, 32, 8064)  float32  — 32 EEG channels, 8064 samples
y        : (n_trials, 4)         float32  — valence, arousal, dominance, liking
subjects : (n_trials,)           int      — participant index per trial
                                            (required for subject-independent
                                            cross-subject evaluation)
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from src.config import (DATA_DIR, N_EEG_CHANNELS, N_TRIALS,
                        N_PARTICIPANTS, SFREQ, RANDOM_SEED)

# DEAP .dat files were pickled with an older NumPy; unpickling them under
# NumPy 2.4 emits a harmless dtype deprecation warning. The data loads
# correctly — silence the noise.
warnings.filterwarnings("ignore", message=".*align.*")

SAMPLES_PER_TRIAL = 63 * SFREQ  # 63 s × 128 Hz = 8064


# ── Public entry point ─────────────────────────────────────────────────────────

def load_deap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect the data format in DATA_DIR and load it. Returns (X, y, subjects)."""
    files = os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []

    dat_files = sorted(f for f in files if f.lower().endswith(".dat"))
    npz_files = sorted(f for f in files if f.lower().endswith(".npz"))
    npy_files = sorted(f for f in files if f.lower().endswith(".npy"))
    csv_files = sorted(f for f in files if f.lower().endswith(".csv"))

    if dat_files:
        X, y, subjects = _load_dat(dat_files)
    elif npz_files:
        X, y, subjects = _load_npz(npz_files[0])
    elif npy_files:
        X, y, subjects = _load_npy(npy_files)
    elif csv_files:
        X, y, subjects = _load_csv(csv_files[0])
    else:
        print("[load_data] No dataset files found — running in DEMO MODE.")
        return _generate_demo_data()

    _validate(X, y, subjects)
    print(f"[load_data] Final dataset → X{X.shape}, y{y.shape}, "
          f"subjects: {len(np.unique(subjects))} unique")
    return X, y, subjects


# ── Format: original DEAP .dat pickle files ────────────────────────────────────

def _load_dat(dat_files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_X, all_y, all_subj = [], [], []
    for subj_idx, fname in enumerate(dat_files):
        with open(os.path.join(DATA_DIR, fname), "rb") as f:
            subject = pickle.load(f, encoding="latin1")
        data   = np.asarray(subject["data"])     # (40, 40, 8064)
        labels = np.asarray(subject["labels"])   # (40, 4)
        all_X.append(data[:, :N_EEG_CHANNELS, :])           # EEG channels only
        all_y.append(labels)
        all_subj.append(np.full(data.shape[0], subj_idx, dtype=int))
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    subjects = np.concatenate(all_subj, axis=0)
    print(f"[load_data] Loaded {len(dat_files)} .dat file(s) "
          f"(DEAP pickle format, 1 subject per file)")
    return X, y, subjects


# ── Format: NumPy .npz archive ─────────────────────────────────────────────────

def _load_npz(fname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arch = np.load(os.path.join(DATA_DIR, fname))
    keys = list(arch.keys())
    data_key  = _pick_key(keys, ["data", "x", "eeg", "signals"])
    label_key = _pick_key(keys, ["labels", "y", "label", "ratings"])
    X = np.asarray(arch[data_key]).astype(np.float32)
    y = np.asarray(arch[label_key]).astype(np.float32)
    X = _fix_channels(X)
    subjects = _subject_ids_for(keys, arch, len(X))
    print(f"[load_data] Loaded {fname} (npz; data='{data_key}', labels='{label_key}')")
    return X, y, subjects


# ── Format: NumPy .npy array pair ──────────────────────────────────────────────

def _load_npy(npy_files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_file  = _pick_file(npy_files, ["data", "x", "eeg", "signal"])
    label_file = _pick_file(npy_files, ["label", "y", "rating"])
    if data_file is None or label_file is None:
        raise ValueError(
            f"[load_data] Found .npy files {npy_files} but could not identify "
            f"which is data and which is labels. Rename them to data.npy and "
            f"labels.npy."
        )
    X = np.load(os.path.join(DATA_DIR, data_file)).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, label_file)).astype(np.float32)
    X = _fix_channels(X)
    subjects = _synthetic_subjects(len(X))
    print(f"[load_data] Loaded {data_file} + {label_file} (npy pair)")
    return X, y, subjects


# ── Format: CSV (common on Kaggle DEAP mirrors) ────────────────────────────────

def _load_csv(fname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    cols_lower = [c.lower().strip() for c in df.columns]

    label_names = ["valence", "arousal", "dominance", "liking"]
    label_idx = [i for i, c in enumerate(cols_lower)
                 if any(name in c for name in label_names)]
    if len(label_idx) < 2:
        raise ValueError(
            f"[load_data] CSV '{fname}' columns {list(df.columns)[:8]}…\n"
            f"  Could not find valence/arousal label columns by name."
        )

    label_cols  = [df.columns[i] for i in label_idx]
    signal_cols = [c for c in df.columns if c not in label_cols]

    y = np.zeros((len(df), 4), dtype=np.float32)
    for lc in label_cols:
        lc_l = lc.lower()
        for k, name in enumerate(label_names):
            if name in lc_l:
                y[:, k] = df[lc].to_numpy(dtype=np.float32)

    signals = df[signal_cols].to_numpy(dtype=np.float32)
    X = _reshape_signal_columns(signals)

    # Use a 'subject'/'participant' column if present, else synthetic
    subj_col = next((c for c in df.columns
                     if c.lower() in ("subject", "participant", "subject_id")), None)
    if subj_col is not None:
        subjects = df[subj_col].astype("category").cat.codes.to_numpy()
    else:
        subjects = _synthetic_subjects(len(X))
    print(f"[load_data] Loaded {fname} (csv; {len(signal_cols)} signal cols)")
    return X, y, subjects


def _reshape_signal_columns(signals: np.ndarray) -> np.ndarray:
    n_trials, n_signal = signals.shape
    if n_signal % N_EEG_CHANNELS == 0:
        samples = n_signal // N_EEG_CHANNELS
        print(f"[load_data] CSV signals reshaped to "
              f"({n_trials}, {N_EEG_CHANNELS}, {samples})")
        return signals.reshape(n_trials, N_EEG_CHANNELS, samples).astype(np.float32)
    raise ValueError(
        f"[load_data] CSV has {n_signal} signal columns, not divisible by "
        f"{N_EEG_CHANNELS}. Looks like a pre-extracted FEATURE CSV — share the "
        f"column layout to switch the pipeline to feature-CSV mode."
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pick_key(keys: list[str], candidates: list[str]) -> str:
    for cand in candidates:
        for k in keys:
            if cand == k.lower() or cand in k.lower():
                return k
    raise ValueError(f"[load_data] Could not match {candidates} in keys {keys}")


def _pick_file(files: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        for f in files:
            if cand in f.lower():
                return f
    return None


def _subject_ids_for(keys, arch, n: int) -> np.ndarray:
    """Use a subject array from an npz archive if present, else synthetic."""
    for cand in ("subjects", "subject", "participant", "groups"):
        for k in keys:
            if cand in k.lower():
                return np.asarray(arch[k]).astype(int)[:n]
    return _synthetic_subjects(n)


def _synthetic_subjects(n: int) -> np.ndarray:
    """
    Assign synthetic subject IDs assuming N_TRIALS consecutive trials per
    subject. Used only when the data has no subject metadata — a warning is
    printed because subject-independent evaluation is then not truly
    cross-subject.
    """
    if n % N_TRIALS == 0:
        subj = np.repeat(np.arange(n // N_TRIALS), N_TRIALS)
    else:
        subj = np.arange(n) // N_TRIALS
    print("[load_data] WARNING — no subject metadata in this format; subject "
          "IDs assigned synthetically (assumes 40 consecutive trials/subject). "
          "Subject-independent results are only valid if this assumption holds.")
    return subj.astype(int)


def _fix_channels(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"[load_data] Expected 3D data array, got {X.shape}")
    if X.shape[1] >= N_EEG_CHANNELS:
        return X[:, :N_EEG_CHANNELS, :].astype(np.float32)
    raise ValueError(f"[load_data] Data has {X.shape[1]} channels, "
                     f"fewer than {N_EEG_CHANNELS} expected.")


def _validate(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> None:
    if X.ndim != 3 or X.shape[1] != N_EEG_CHANNELS:
        raise ValueError(f"[load_data] X shape {X.shape}, expected (n,{N_EEG_CHANNELS},samples)")
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"[load_data] y shape {y.shape}, expected (n,>=2)")
    if not (X.shape[0] == y.shape[0] == subjects.shape[0]):
        raise ValueError(f"[load_data] count mismatch: X{X.shape[0]} "
                         f"y{y.shape[0]} subjects{subjects.shape[0]}")


# ── Demo data ──────────────────────────────────────────────────────────────────

def _generate_demo_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic EEG-like data with correct shapes for pipeline testing."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_PARTICIPANTS * N_TRIALS
    X = rng.standard_normal((n_total, N_EEG_CHANNELS, SAMPLES_PER_TRIAL)).astype(np.float32)
    t = np.linspace(0, 63, SAMPLES_PER_TRIAL)
    for freq in [2, 6, 10, 20, 35]:
        X += 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    y = rng.integers(1, 10, size=(n_total, 4)).astype(np.float32)
    subjects = np.repeat(np.arange(N_PARTICIPANTS), N_TRIALS)
    print(f"[load_data] DEMO — synthetic X{X.shape}, y{y.shape}, "
          f"{N_PARTICIPANTS} subjects")
    return X, y, subjects

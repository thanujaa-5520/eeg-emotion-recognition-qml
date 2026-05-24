"""
All hyperparameters and paths in one place.
Change values here to tune the experiment.

Quick-test mode: set the environment variable EEG_QML_QUICK=1 to run a fast
end-to-end smoke test (few epochs, subsampled data). Used for verifying the
pipeline; use the normal mode for real experiments and paper results.
"""
import os

QUICK = os.environ.get("EEG_QML_QUICK", "0") == "1"

# FRESH=1 ignores any existing progress.jsonl and starts the benchmark over.
# Default (0) resumes: completed model-runs are skipped on restart.
FRESH = os.environ.get("EEG_QML_FRESH", "0") == "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# ── Dataset ────────────────────────────────────────────────────────────────────
SFREQ           = 128        # DEAP sampling frequency (Hz)
N_EEG_CHANNELS  = 32        # EEG channels only (drop 8 peripheral)
N_TRIALS        = 40        # trials per participant
N_PARTICIPANTS  = 32
LABEL_THRESHOLD = 5.0       # ≥5 → high (1), <5 → low (0)
VALENCE_COL     = 0         # column index in DEAP label array
AROUSAL_COL     = 1

# ── Frequency bands (Hz) ───────────────────────────────────────────────────────
FREQ_BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ── Feature extraction ─────────────────────────────────────────────────────────
PCA_COMPONENTS = 16         # PCA output dim = quantum circuit input dim

# ── Train / test ───────────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── Classical baselines ────────────────────────────────────────────────────────
SVM_C      = 1.0
SVM_GAMMA  = "scale"
RF_TREES   = 100

# ── EEGNet (PyTorch CNN baseline) ──────────────────────────────────────────────
EEGNET_EPOCHS    = 4 if QUICK else 50
EEGNET_LR        = 1e-3
EEGNET_BATCH     = 32

# ── Hybrid quantum-classical model ─────────────────────────────────────────────
N_QUBITS         = 8        # must equal PCA_COMPONENTS / 2, kept ≤ PCA_COMPONENTS
VQC_LAYERS       = 2        # number of variational layers in the quantum circuit
QUANTUM_EPOCHS   = 3 if QUICK else 30
QUANTUM_LR       = 0.01
QUANTUM_BATCH    = 32

# ── Quick-test subsampling ─────────────────────────────────────────────────────
QUICK_MAX_TRIALS = 240      # cap total trials in quick mode for a fast smoke test

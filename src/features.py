"""
Step 3 — Feature Extraction.

Welch PSD → 5 frequency-band powers per channel → 160-feature vector per trial.
StandardScaler + PCA reduce to PCA_COMPONENTS (16) for the quantum circuit and
classical (SVM / RF) inputs.

Leakage control: the scaler and PCA are fit on the TRAIN split only, then used
to transform validation and test. EEGNet uses raw EEG (not these features).
"""
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.config import SFREQ, FREQ_BANDS, PCA_COMPONENTS, RANDOM_SEED


def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    return psd[mask].mean() if mask.any() else 0.0


def extract_psd_features(X: np.ndarray) -> np.ndarray:
    """
    X : (n_trials, n_channels, n_samples)
    returns : (n_trials, n_channels * n_bands)  =  (n, 160)
    """
    n_trials, n_ch, _ = X.shape
    n_bands = len(FREQ_BANDS)
    features = np.zeros((n_trials, n_ch * n_bands), dtype=np.float32)

    for i, trial in enumerate(X):
        row = []
        for ch in range(n_ch):
            freqs, psd = welch(trial[ch], fs=SFREQ, nperseg=SFREQ * 2)
            for fmin, fmax in FREQ_BANDS.values():
                row.append(_band_power(psd, freqs, fmin, fmax))
        features[i] = row
    return features


def apply_pca(train_feats: np.ndarray,
              val_feats: np.ndarray,
              test_feats: np.ndarray) -> tuple:
    """
    Fit StandardScaler + PCA on TRAIN features only, transform all three splits.
    Returns (train_pca, val_pca, test_pca, pca_object).
    """
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_feats)
    val_s   = scaler.transform(val_feats)
    test_s  = scaler.transform(test_feats)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    train_p = pca.fit_transform(train_s).astype(np.float32)
    val_p   = pca.transform(val_s).astype(np.float32)
    test_p  = pca.transform(test_s).astype(np.float32)

    var = pca.explained_variance_ratio_.sum() * 100
    print(f"[features] PCA {PCA_COMPONENTS} components explain {var:.1f}% "
          f"of train variance (fit on train only)")
    return train_p, val_p, test_p, pca


def extract_features(splits: dict) -> dict:
    """
    Extract PSD features for train/val/test, fit PCA on train, add reduced
    features back into the splits dict. Raw EEG splits are kept for EEGNet.
    """
    print("[features] Extracting PSD features — train …")
    train_feats = extract_psd_features(splits["X_train"])
    print("[features] Extracting PSD features — val …")
    val_feats   = extract_psd_features(splits["X_val"])
    print("[features] Extracting PSD features — test …")
    test_feats  = extract_psd_features(splits["X_test"])
    print(f"[features] Raw feature shape: {train_feats.shape[1]} per trial")

    train_p, val_p, test_p, pca = apply_pca(train_feats, val_feats, test_feats)
    splits["X_train_pca"] = train_p   # for SVM, RF, Quantum
    splits["X_val_pca"]   = val_p
    splits["X_test_pca"]  = test_p
    splits["pca"]         = pca
    return splits

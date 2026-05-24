# Hybrid Quantum-Classical EEG Emotion Recognition — Results Report
Comparative study of four models for EEG-based valence and arousal classification on the DEAP dataset, evaluated under two protocols.
## 1. Methodology Summary
- **Dataset:** DEAP — 32 participants x 40 trials = 1280 trials, 32 EEG channels, 128 Hz, 63 s per trial.
- **Tasks:** binary valence and binary arousal (rating >= 5 -> high, < 5 -> low).
- **Preprocessing:** per-trial per-channel z-score normalisation. DEAP data is already band-pass filtered (4-45 Hz).
- **Features (SVM / RF / HybridQNN):** Welch PSD power in 5 bands (delta, theta, alpha, beta, gamma) x 32 channels = 160 features, reduced to 16 via PCA (fit on train only).
- **EEGNet** consumes raw EEG time-series directly.
- **HybridQNN:** classical encoder -> 8-qubit variational quantum circuit (2 layers, PennyLane default.qubit simulator) -> classical head.
- **Training:** EEGNet 50 epochs, HybridQNN 30 epochs, both with best-validation checkpointing. SVM and RF are single-shot fits.
- **Protocols:** (a) *subject-dependent* — stratified random 70/10/20 split; (b) *subject-independent* — 22/4/6 participants held out so test subjects never appear in training.

## 2. Results — Subject-Dependent Protocol
| Model | Task | Acc % | Prec % | Rec % | F1 % | ROC-AUC | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|---|
| SVM | valence | 56.25 | 58.13 | 81.38 | 67.82 | 0.5727 | 0.07 | 48.4 |
| SVM | arousal | 62.89 | 65.1 | 81.7 | 72.46 | 0.6448 | 0.03 | 25.6 |
| RandomForest | valence | 61.72 | 63.74 | 75.17 | 68.99 | 0.6318 | 1.03 | 128.7 |
| RandomForest | arousal | 61.33 | 65.7 | 73.86 | 69.54 | 0.6342 | 0.28 | 70.9 |
| EEGNet | valence | 54.69 | 59.6 | 62.07 | 60.81 | 0.5555 | 3073.99 | 4694.7 |
| EEGNet | arousal | 60.16 | 64.41 | 74.51 | 69.09 | 0.599 | 3117.93 | 5546.2 |
| HybridQNN | valence | 55.86 | 58.08 | 79.31 | 67.06 | 0.5358 | 52.4 | 111.2 |
| HybridQNN | arousal | 61.33 | 62.98 | 85.62 | 72.58 | 0.599 | 53.12 | 110.6 |

## 3. Results — Subject-Independent Protocol
| Model | Task | Acc % | Prec % | Rec % | F1 % | ROC-AUC | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|---|
| SVM | valence | 50.42 | 60.17 | 49.65 | 54.41 | 0.5217 | 0.06 | 32.8 |
| SVM | arousal | 59.58 | 66.48 | 77.07 | 71.39 | 0.5168 | 0.02 | 23.8 |
| RandomForest | valence | 45.83 | 57.47 | 34.97 | 43.48 | 0.493 | 0.5 | 65.9 |
| RandomForest | arousal | 57.5 | 64.86 | 76.43 | 70.18 | 0.5071 | 0.29 | 91.8 |
| EEGNet | valence | 49.17 | 59.63 | 45.45 | 51.59 | 0.512 | 3469.57 | 8645.6 |
| EEGNet | arousal | 57.08 | 68.0 | 64.97 | 66.45 | 0.522 | 3352.43 | 6490.3 |
| HybridQNN | valence | 59.58 | 59.58 | 100.0 | 74.67 | 0.5727 | 52.99 | 112.2 |
| HybridQNN | arousal | 52.08 | 64.79 | 58.6 | 61.54 | 0.4558 | 48.66 | 107.2 |

## 4. Combined Comparison
See `metrics_combined.csv`, `comparison_subject_dependent.png`, `comparison_subject_independent.png`, and `generalization_gap.png`.

## 5. Statistical Observations
- **Subject-Dependent:** mean accuracy across all models/tasks = 59.28%. Highest non-degenerate single result: SVM on arousal (62.89%).
- **Subject-Independent:** mean accuracy across all models/tasks = 53.91%. Highest non-degenerate single result: SVM on arousal (59.58%).
- **Degenerate-predictor caution.** The following run(s) collapsed to predicting a single class for every test sample (recall ~100% or ~0%). Their accuracy reflects the class prior, NOT genuine discrimination, and must not be read as a performance result:
  - HybridQNN / valence / subject_independent — recall 100.0%, accuracy 59.58%, F1 74.67% (ROC-AUC 0.5727 — the underlying scores carry only weak signal; the decision threshold collapsed). Any aggregate that includes this run is correspondingly inflated.
- The hybrid quantum-classical model reached 58.59% (subject-dependent) and 55.83% (subject-independent) mean accuracy (note: the subject-independent figure is inflated by a collapsed-predictor run — see the caution above). On the subject-dependent protocol it performed competitively with the classical baselines; it did not consistently outperform them, and on several task/protocol combinations a classical model scored higher.
- Reported differences between models are modest and, given the small test sets, should be interpreted as broadly comparable rather than as decisive rankings. No model dominates universally.

## 6. Generalization Analysis
Performance under the subject-independent protocol is expected to be lower than under the subject-dependent protocol for all models. Per-model mean-accuracy comparison:

| Model | Subject-Dependent | Subject-Independent | Gap |
|---|---|---|---|
| SVM | 59.57% | 55.00% | +4.57% |
| RandomForest | 61.52% | 51.66% | +9.86% |
| EEGNet | 57.42% | 53.12% | +4.30% |
| HybridQNN | 58.59% | 55.83% | +2.77% |

*Caveat:* the HybridQNN subject-independent mean — and therefore its apparently small generalization gap — is inflated by a collapsed-predictor run (Section 5). Its gap should not be interpreted as superior cross-subject generalization.

**Why subject-independent performance is lower:** EEG signals exhibit substantial inter-subject variability — differences in skull anatomy, electrode placement, baseline neural activity and individual emotional expression. In the subject-dependent protocol a model can exploit participant-specific patterns because trials from each participant appear in both training and test. In the subject-independent protocol the model must generalise to entirely unseen participants, which is a harder and more realistic evaluation. The gap reflects this difficulty rather than a defect in any model, and is consistent with the EEG emotion-recognition literature.

## 7. Runtime Comparison
| Model | Protocol | Mean Train (s) | Mean Infer (ms) |
|---|---|---|---|
| SVM | Subject-Dependent | 0.05 | 37.0 |
| SVM | Subject-Independent | 0.04 | 28.3 |
| RandomForest | Subject-Dependent | 0.66 | 99.8 |
| RandomForest | Subject-Independent | 0.40 | 78.8 |
| EEGNet | Subject-Dependent | 3095.96 | 5120.4 |
| EEGNet | Subject-Independent | 3411.00 | 7568.0 |
| HybridQNN | Subject-Dependent | 52.76 | 110.9 |
| HybridQNN | Subject-Independent | 50.83 | 109.7 |

## 8. Reproducibility Notes
- Global random seed: **42** (Python, NumPy, PyTorch).
- PCA and StandardScaler are fit on the training split only.
- The same train/val/test split is reused across all four models within a protocol, ensuring a fair comparison.
- **Subject-Independent** subject assignment — train: [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20, 21, 23, 24, 25, 29, 30]; val: [5, 10, 22, 28]; test: [7, 18, 19, 26, 27, 31].
- Quantum circuit executed on the PennyLane `default.qubit` statevector simulator (deterministic, no shot noise).

## 9. Limitations
- The quantum component runs on a noiseless simulator; results do not account for real quantum-hardware noise or decoherence.
- The quantum circuit is intentionally compact (8 qubits, 2 layers); larger circuits were not explored.
- PCA to 16 components discards some variance before the SVM / RF / quantum models; EEGNet avoids this by using raw EEG.
- The subject-independent protocol uses a single fixed 22/4/6 split rather than full leave-one-subject-out cross-validation, for computational tractability on CPU.
- Binary thresholding at rating = 5 follows common practice but discards the intensity information in the original 1-9 scale.

## 10. Future Work
- Evaluate with full leave-one-subject-out cross-validation.
- Test the hybrid model on real quantum hardware or noisy simulators to assess robustness.
- Explore alternative quantum feature encodings and circuit depths under controlled comparison.
- Extend from binary to multi-class (e.g. four-quadrant valence-arousal) classification.
- Investigate subject-adaptive or transfer-learning strategies to narrow the subject-independent generalization gap.

---
*Generated automatically by the experiment pipeline. Wording is intentionally cautious; the hybrid quantum-classical model is presented as a promising architecture showing competitive performance under the evaluated conditions, not as a universally superior method.*

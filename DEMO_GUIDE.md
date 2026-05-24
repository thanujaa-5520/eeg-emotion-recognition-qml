# DEMO GUIDE — Hybrid Quantum-Classical EEG Emotion Recognition
*Keep this file open during your Google Meet. Sections 5 and 7 are what you read/use live.*

---

## PART 1 — PROJECT FILE MAP (what each file does)

Project folder: `EEG_Emotion_QML/`

| File | What it does |
|---|---|
| `main.py` | **Main script.** Runs the whole experiment: loads data → preprocesses → trains all 4 models under both protocols → saves results. |
| `src/config.py` | All settings in one place — file paths, frequency bands, number of qubits, epochs, random seed. |
| `src/load_data.py` | **Dataset loading.** Reads the 32 DEAP `.dat` files and assembles the EEG data, labels, and subject IDs. |
| `src/preprocess.py` | **Preprocessing + splitting.** Normalises the EEG, builds binary labels, and creates the train/validation/test splits for both protocols. |
| `src/features.py` | **Feature extraction.** Converts raw EEG into frequency-band power features (PSD) and reduces them with PCA. |
| `src/classical.py` | **Three models:** SVM, Random Forest, and EEGNet (the deep-learning CNN baseline). |
| `src/quantum_model.py` | **Hybrid Quantum-Classical model** — the variational quantum circuit built with PennyLane. |
| `src/evaluate.py` | Computes metrics (accuracy, F1, ROC-AUC…) and draws confusion matrices, ROC curves, training curves. |
| `src/report.py` | **Report generation.** Builds the CSV tables, comparison charts, and `RESULTS_REPORT.md`. |
| `phase1_validation.py` | A one-off data-validation script (checks the dataset for errors before training). |
| `regenerate_report.py` | Rebuilds the report from saved results without re-training. |
| `data/` | The 32 DEAP `.dat` files (one per participant). |
| `models/` | The 4 trained model checkpoints (`.pt` files). |
| `results/` | **All outputs** — metrics CSVs, plots, prediction files, and the report. |

> Note: SVM, Random Forest and EEGNet all live in `classical.py`. Only the quantum model is in `quantum_model.py`.

---

## PART 2 — HOW THE CODE WORKS (beginner-friendly)

**1. Loading the DEAP dataset.**
DEAP gives 32 files, one per participant (`s01.dat` … `s32.dat`). `load_data.py` opens each file, stacks them together, and also records which participant each recording came from.

**2. The input shape — `X(1280, 32, 8064)` and `y(1280, 4)`.**
- **1280 trials** = 32 participants × 40 video clips each. Each "trial" is one person watching one music video.
- **32 EEG channels** = 32 electrodes placed on the scalp, each recording brain electrical activity.
- **8064 time points** = 63 seconds of recording × 128 samples per second.
- **4 labels** = the person's self-rating of valence, arousal, dominance, liking. We use **valence** and **arousal**.

**3. Creating valence and arousal labels.**
Participants rated each video 1–9. We make it a binary problem: rating **≥ 5 → "high" (1)**, **< 5 → "low" (0)**. So "valence" becomes high/low, and "arousal" becomes high/low.

**4. Preprocessing.**
Each EEG channel is **z-score normalised** per trial — rescaled to mean 0, standard deviation 1 — so all channels are on the same scale. (DEAP data is already noise-filtered to 4–45 Hz by the dataset authors, so we don't re-filter.)

**5. Feature extraction + PCA.**
For SVM, Random Forest and the quantum model, we convert each EEG trial into **frequency-band powers**: how much energy is in the delta, theta, alpha, beta, gamma bands, for each of 32 channels = **160 features**. Then **PCA** compresses those 160 numbers down to **16** — keeping the most informative directions. (EEGNet skips this — it learns directly from the raw EEG.)

**6. Train / validation / test split.**
- **Train** — the model learns from this.
- **Validation** — used to pick the best version of the model during training (best checkpoint).
- **Test** — held back; only used at the very end to measure true performance.

**7. The two protocols.**
- **Subject-dependent:** trials are split randomly. The same person can appear in both training and test. Easier — the model can learn person-specific patterns.
- **Subject-independent:** whole participants are held out. 22 people for training, 4 for validation, 6 for testing. The model is tested on people it has **never seen**. Harder, but more realistic and more rigorous.

**8. Leakage checking.**
"Subject leakage" = the same person appearing in both training and test, which would make results look better than they really are. In the subject-independent split, the code checks that the three groups of participants have **no overlap** and stops with an error if they do. The run confirmed: `leakage check: NONE — clean`.

---

## PART 3 — HOW EACH MODEL WAS TRAINED

| Model | Input | Training method | Output | Metrics |
|---|---|---|---|---|
| **SVM** | 16 PCA features | Support Vector Machine with an RBF kernel — one-shot fit on the training set | Class prediction + decision score | Accuracy, Precision, Recall, F1, ROC-AUC, train/inference time |
| **Random Forest** | 16 PCA features | 100 decision trees, majority vote — one-shot fit | Class prediction + probability | Same as above |
| **EEGNet** | Raw EEG `(32 × 8064)` | A compact CNN trained for **50 epochs** with the Adam optimiser; the **best-validation-accuracy** version is kept | Class prediction + probability + per-epoch training curve | Same + training curves |
| **HybridQNN** | 16 PCA features | A classical layer feeds an **8-qubit variational quantum circuit** (PennyLane simulator), trained **30 epochs**, best-validation checkpoint kept | Class prediction + probability + training curve | Same + training curves |

Every model is trained on the **same train set** and evaluated on the **same test set** within each protocol — so the comparison is fair.

---

## PART 4 — HOW THE OUTPUTS WERE GENERATED

After each model is tested, `evaluate.py` and `report.py` turn the raw predictions into:

- **`metrics_subject_dependent.csv` / `metrics_subject_independent.csv`** — one row per model per task, with all metrics, for each protocol.
- **`metrics_combined.csv`** — all results from both protocols in one file.
- **Confusion matrices** (`cm_*.png`) — a 2×2 grid showing correct vs incorrect predictions (low/high).
- **ROC curves** (`roc_*.png`) — plot of true-positive vs false-positive rate; the area under it (AUC) summarises how well the model separates the classes.
- **Training curves** (`training_*.png`) — loss and accuracy at each epoch for EEGNet and the quantum model, with the best epoch marked.
- **`RESULTS_REPORT.md` / `.docx`** — `report.py` assembles everything into a full report: methodology, both results tables, statistical observations, generalization analysis, runtime, limitations, future work.

The chain is: **predictions → metrics → tables + plots → report.**

---

## PART 5 — GOOGLE MEET SPEAKING SCRIPT
*Read this naturally; pause to share your screen where indicated.*

**(1) Introduction**
"Good morning. My project is **EEG-based emotion recognition using a hybrid quantum-classical machine learning model**. The goal is to predict a person's emotional state — specifically valence and arousal — from their brain signals, and to compare a quantum-enhanced model against standard classical machine-learning models. The novel part is that this is the first time a hybrid quantum-classical model has been applied to EEG *emotion* recognition — earlier quantum-EEG work only covered seizure detection and motor imagery."

**(2) Dataset**
"I used the **DEAP dataset**, a standard public dataset for EEG emotion research. It has 32 participants who each watched 40 music videos while their brain activity was recorded with 32 EEG electrodes. That gives 1280 recordings. After each video, participants rated their emotion. I converted those ratings into two binary classification tasks — high vs low valence, and high vs low arousal."

**(3) Code walkthrough** *(share VS Code)*
"The project is organised into modules. `load_data.py` reads the dataset; `preprocess.py` normalises the signals and creates the splits; `features.py` extracts frequency-band features and applies PCA; `classical.py` has the SVM, Random Forest and EEGNet models; `quantum_model.py` has the hybrid quantum model; and `main.py` runs the whole experiment end to end."

**(4) Model training**
"I trained four models. SVM and Random Forest are classical baselines using frequency-band features. EEGNet is a deep-learning CNN that learns from the raw EEG. The hybrid model uses a classical layer feeding an 8-qubit variational quantum circuit built in PennyLane. I evaluated all four under two protocols — subject-dependent, where the split is random, and subject-independent, where whole participants are held out so the model is tested on people it has never seen."

**(5) Results** *(share RESULTS_REPORT.docx)*
"Under the subject-dependent protocol, all four models scored in the mid-50s to low-60s percent range. The hybrid quantum model was **competitive** with the classical models — for example it matched Random Forest on arousal — but it did not outperform them across the board. Under the harder subject-independent protocol, accuracy dropped for every model, which is expected, because generalising to unseen people is difficult."

**(6) Important limitation — be honest here**
"I want to be transparent about one limitation. In the subject-independent valence task, the hybrid quantum model **collapsed into majority-class prediction** — it predicted 'high' for every test sample. Its accuracy there just reflects the class proportion, not real learning. I detected this from the recall being 100%, and I've flagged it explicitly in my report rather than presenting it as a result. This kind of collapse can happen with weak features on a hard task, and reporting it honestly is part of good scientific practice."

**(7) Conclusion**
"In conclusion, I built a complete, reproducible benchmarking pipeline and showed that a hybrid quantum-classical model can perform **competitively** with classical models for EEG emotion recognition — though not better, and with one clear failure case I've documented. The contribution is being the first to apply hybrid quantum machine learning to this specific task, and providing an honest, rigorous comparison under both evaluation protocols. Future work includes testing on real quantum hardware and using leave-one-subject-out cross-validation."

---

## PART 6 — COMMANDS USED
*(Windows PowerShell. `python` here is Python 3.14.)*

**1. Environment** — Python 3.14 was already installed.

**2. Install dependencies:**
```
pip install numpy pandas scipy scikit-learn matplotlib torch pennylane pennylane-qiskit kaggle
```

**3. Quick smoke test** (fast check — few epochs, small subset):
```
$env:EEG_QML_QUICK="1"; python main.py
```

**4. Full publication run** (both protocols, all 4 models, 50-epoch EEGNet):
```
$env:EEG_QML_QUICK="0"; $env:EEG_QML_FRESH="1"; python main.py
```

**5. Regenerate the report** (rebuild tables/plots/report without re-training):
```
python regenerate_report.py
```

**6. Resume after an interruption** (skips already-finished models automatically):
```
python main.py
```

---

## PART 7 — TEACHER Q&A PREPARATION

**Q: Why the DEAP dataset?**
It is the standard, widely-used public dataset for EEG emotion recognition. Using it means my results can be compared with existing literature.

**Q: Why valence and arousal?**
Psychology models emotion on two axes — valence (how positive/negative) and arousal (how calm/excited). Together they describe most emotional states, and DEAP provides ratings for both.

**Q: Why use EEGNet?**
EEGNet is a well-known, compact deep-learning model designed specifically for EEG. It's a strong, fair deep-learning baseline to compare the quantum model against.

**Q: Why use quantum machine learning?**
Quantum circuits process information in a very high-dimensional space, which in theory can capture complex patterns. I wanted to test whether a hybrid quantum-classical model offers any benefit for EEG emotion recognition — a question nobody had studied before.

**Q: What is subject leakage?**
When data from the same person appears in both the training and test sets. The model can then "recognise the person" instead of learning emotion, making results look better than they truly are.

**Q: Why is subject-independent evaluation important?**
Because in real use, an emotion-recognition system meets new people it was never trained on. Subject-independent testing measures that real-world ability and is more rigorous and credible.

**Q: Why did the quantum model collapse?**
On the hardest task (cross-subject valence), the features carried very weak signal. The model found that always predicting the majority class minimised its training loss, so it stopped discriminating. I detected it (recall = 100%) and reported it honestly instead of hiding it.

**Q: Is this still publishable?**
Yes. The contribution is the **first application** of hybrid quantum machine learning to EEG emotion recognition, plus a rigorous dual-protocol comparison. Honest reporting of a limitation strengthens credibility — negative and partial results are valid science.

**Q: What is the novelty of this work?**
No prior study applied a hybrid quantum-classical model to EEG **emotion** recognition — earlier quantum-EEG work was only on seizures and motor imagery. I also provide a fair, leakage-free, reproducible benchmark against three classical models under two evaluation protocols.

**Q: Could the quantum model be improved?**
Yes — with better feature encoding, a deeper circuit, class-balancing, or threshold tuning. That is part of my future work.

---
*Tip: speak slowly, share your screen when you reach the code and the report, and don't be afraid of the limitation — explaining it well shows scientific maturity.*

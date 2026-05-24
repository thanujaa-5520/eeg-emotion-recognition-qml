# Project Structure

```
EEG_Emotion_QML/
│
├── src/                          # Core Python modules
│   ├── config.py                 # All hyperparameters and paths
│   ├── load_data.py              # DEAP dataset loader (multi-format)
│   ├── preprocess.py             # Per-trial z-score normalisation + train/val/test splits
│   ├── features.py               # Welch PSD feature extraction + PCA reduction
│   ├── classical.py              # SVM, Random Forest, and EEGNet (CNN) models
│   ├── quantum_model.py          # Hybrid Quantum-Classical model (PennyLane + PyTorch)
│   ├── evaluate.py               # Metrics, confusion matrices, ROC curves
│   ├── report.py                 # Auto-generates RESULTS_REPORT.md and comparison plots
│   └── __init__.py
│
├── data/                         # [NOT INCLUDED] DEAP .dat files (s01–s32)
│                                 # Download from: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
│
├── models/                       # Saved model checkpoints (.pt files)
│   ├── eegnet_arousal.pt         # Best EEGNet checkpoint for arousal
│   ├── eegnet_valence.pt         # Best EEGNet checkpoint for valence
│   ├── quantum_arousal.pt        # Best HybridQNN checkpoint for arousal
│   └── quantum_valence.pt        # Best HybridQNN checkpoint for valence
│
├── results/                      # All outputs generated after training
│   ├── RESULTS_REPORT.md         # Full auto-generated results report
│   ├── metrics_combined.csv      # Consolidated metrics table
│   ├── metrics_subject_dependent.csv
│   ├── metrics_subject_independent.csv
│   ├── comparison_subject_dependent.png
│   ├── comparison_subject_independent.png
│   ├── generalization_gap.png
│   ├── cm_*.png                  # Confusion matrices for each model/task/protocol
│   ├── roc_*.png                 # ROC curves (combined per task/protocol)
│   ├── training_*.png            # Training curves (EEGNet & HybridQNN)
│   ├── pred_*.npz                # Raw predictions saved for each model run
│   └── progress.jsonl            # Crash-resilient run checkpoint log
│
├── main.py                       # Main pipeline — trains all 4 models, evaluates, generates report
├── phase1_validation.py          # Phase 1 standalone validation script
├── regenerate_report.py          # Re-generate report from existing results without re-training
├── DEMO_GUIDE.md                 # Step-by-step demo and presentation guide
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excluded files (data/, node_modules/, __pycache__, etc.)
├── LICENSE                       # MIT License
└── project_structure.md          # This file
```

## Key Design Decisions

| Decision | Reason |
|---|---|
| `data/` excluded from repo | DEAP files are ~3.1 GB total; must be downloaded separately |
| `models/` included | Checkpoints are only ~128 KB total — small enough to version |
| `results/` included | All plots and CSVs are lightweight outputs; useful for reproduction verification |
| Crash-resilient pipeline | `progress.jsonl` lets the pipeline resume after interruptions without losing completed runs |
| Two evaluation protocols | Subject-dependent gives upper-bound accuracy; subject-independent tests real-world generalisation |

# SoccerNet Person Re-Identification - Biometric Systems

This repository contains the implementation and evaluation of various person re-identification (Re-ID) strategies on the [SoccerNet Re-ID v3 dataset](https://www.soccer-net.org/), with a specific focus on **Biometric System Analysis**.

##  Project Overview

The project explores person re-identification in broadcast soccer footage, treating it as an **Closed-Set Identification** problem. We implement several state-of-the-art architectures and evaluate them using unconventional biometric metrics like DIR (Detect and Identification Rate) and SRR (System Response Reliability).

### Key Features
- **Multi-Model Pipeline**: Implementation of DINOv2 (ViT), ResNet-50-IBN (CNN), and OsNet-AIN (Optimized Re-ID).
- **Advanced Ensembles**: Feature concatenation, Distance averaging, and Rank-level fusion (Borda Count).
- **Biometric Calibration**: Calculation of EER (Equal Error Rate), DET curves, and Margin of Error $M(t)$.
- **Qualitative Analysis**: Automated identification of "Goats" (challenging queries) using automated SRR ranking and Doddington Zoo taxonomy.
- **Standalone Biometric Tool**: A dedicated script for post-processing distance matrices to extract biometric insights.

##  Results Summary

The following table summarizes our findings on the SoccerNet v3 Validation Set:

| Method | mAP | Rank-1 | SRR | EER | DIR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OsNet-AIN (Best Single)** | **56.83%** | **43.64%** | **0.0081** | **7.48%** | **8.67%** |
| Weighted Ensemble | 55.48% | 43.52% | N/A | 7.94% | **11.90%** |
| Re-Rank Aggressive | 55.08% | 43.08% | 0.0830 | 8.37% | 11.71% |

##  Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- `numpy`, `pillow`, `scikit-learn`
- `torchreid` (included in `sn-reid/`)

### Evaluation
To run the full evaluation pipeline:
```bash
python sn-reid/experiment.py \
  --models sn-reid/final_models/RESNET.tar sn-reid/final_models/DINO.tar sn-reid/final_models/OsNet.tar \
  --archs resnet50_fc512 dinov2_vits14_lora osnet_ain_x1_0 \
  --save-dist
```

### Biometric Analysis Tool
To calculate DIR and Margin metrics from a saved distance matrix:
```bash
python sn-reid/calculate_metrics_standalone.py \
  --dist dist_matrix_osnet_ain_x1_0.npy \
  --pids_q pids_q.npy \
  --pids_g pids_g.npy
```

##  Documentation
A comprehensive technical report [Report_BiometricSystems.pdf](Report_BiometricSystems.pdf) is available, covering:
- **Soft Biometrics Analysis**: Evaluating re-identification against the 5 fundamental biometric requirements.
- **Doddington Zoo Classification**: Characterizing Lambs, Wolves, and Goats in the dataset.
- **Tactical Biometrics**: Perspectives on Spoofing (identical twins) and Camouflage (protective masks) in sports.
- **Template Updating**: Proposals for combating intra-match biometric aging.

##  Contributors
- Alessandro Gautieri
- Michelangelo Crea

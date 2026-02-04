# ⚽ SoccerNet Re-Identification Challenge

[![SoccerNet](https://img.shields.io/badge/SoccerNet-ReID-green.svg)](https://www.soccer-net.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

This repository contains the official codebase for our **SoccerNet Re-Identification (ReID)** project.
It features a unified framework for training state-of-the-art models (ResNet-50, OsNet-AIN, DINOv2), evaluating ensembles, and visualizing results with an interactive dashboard.

---

##  Project Structure

The repository is organized to separate the core library, training benchmarks, and evaluation tools:

```
soccernet/sn-reid/
├── torchreid/              # Custom Deep Learning Library (Core ReID logic)
├── benchmarks/             # Training Scripts & Baseline Configs
│   └── baseline/
│       ├── main.py         # Main training entry point
│       └── configs/        # Hyperparameter configurations (YAML)
├── experiment.py           # Evaluation Suite (Single Models, Ensembles, Re-ranking)
├── gradio_demo.py          # Interactive Demonstration Interface
├── generate_charts.py      # Report Figure Generator
├── final_models/           # Directory for Saved Model Checkpoints
└── datasets/               # Dataset Directory (SoccerNet-v3)
```

---

##  Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
cd sn-reid

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install the TorchReID library in development mode
python setup.py develop
```

### 2. Dataset Download

We use the **SoccerNet-v3** dataset. You can download the ReID split using the official SoccerNet downloader:

```bash
# Install the SoccerNet API
pip install SoccerNet

# Run this python one-liner to download train, valid, test, and challenge splits
python -c "from SoccerNet.Downloader import SoccerNetDownloader; SoccerNetDownloader(LocalDirectory='datasets/soccernetv3').downloadDataTask(task='reid', split=['train', 'valid', 'test', 'challenge'])"
```


---

##  Workflows

###  Training Models

Train specific architectures using the configuration files in `benchmarks/baseline/configs/`.

**Example: Training OsNet-AIN (Best Model)**
```bash
python benchmarks/baseline/main.py \
    --config-file benchmarks/baseline/configs/osnet_ain_x1_0_config.yaml \
    --root datasets
```

**Example: Training ResNet-50**
```bash
python benchmarks/baseline/main.py \
    --config-file benchmarks/baseline/configs/baseline_config.yaml \
    --root datasets \
    model.name resnet50 \
    model.pretrained True
```

**Example: Training DINOv2 (Vision Transformer)**
We provide a standalone script for DINOv2 fine-tuning with LoRA, optimized for consumer GPUs.
```bash
python benchmarks/dino/train.py \
    --root datasets \
    --lr 0.0005 \
    --batch-size 64 \
    --gpu-id 0
```


---

###  Evaluation & Experiments

Use `experiment.py` to evaluate trained models. This script supports **Ensemble** methods and **Re-Ranking** strategies.

**Evaluate a Single Model:**
```bash
python experiment.py --models final_models/OsNet.tar --archs osnet_ain_x1_0
```

**Evaluate an Ensemble (e.g., OsNet + ResNet):**
```bash
python experiment.py \
    --models final_models/OsNet.tar final_models/ResNet.pth \
    --archs osnet_ain_x1_0 resnet50_fc512
```

---

###  Interactive Visualization

Launch the **Gradio Dashboard** to explore the model's performance visually.

```bash
python gradio_demo.py
```

**Features:**
-  **Web Interface**: Opens at `http://localhost:7860`
-  **Query Selection**: Dropdown with "Smart Filtering" to find interesting cases
-  **Visual Results**: Instantly see Query vs. Top-10 Gallery matches
-  **Feedback**: Green borders for correct matches, Red for incorrect

To save visualization examples directly to disk without opening the web UI:
```bash
python gradio_demo.py --save-samples 10 --no-gradio
```

---

###  Report Generation

Generate the charts used in our final report (Bar Plots, CMC Curves):

```bash
python generate_charts.py
```
This will create high-quality PNG figures in the `figures/` directory.

---

##  Authors
- **Crea Michelangelo 1993024**
- **Gautieri Alessandro 2041850**


*Sapienza University of Rome - Computer Vision Project 2025/2026*

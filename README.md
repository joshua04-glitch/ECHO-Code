# Echo-Code
Multi-Task Deep Learning for Automated Cardiac Geometric Feature Extraction from Echocardiography

Author: Josh Thaosatien  
Stanford University — MATSCI 176 Final Project

## Overview

Echo-Code is a deep learning pipeline that produces an automated hemodynamic report from an echocardiographic video. The system integrates three models:

1. Four-chamber segmentation (Attention U-Net)
2. Ejection Fraction regression (ConvNeXt + Transformer)
3. Image quality grading (EfficientNet-B0)

The pipeline extracts cardiac geometry and hemodynamic metrics including:

- Ejection Fraction (EF)
- End-diastolic volume (EDV)
- End-systolic volume (ESV)
- Stroke Volume
- Cardiac Output
- Chamber areas
- Image quality grade

The project demonstrates how heterogeneous partially labeled datasets can be combined to build a unified cardiac analysis pipeline.

---

## Repository Structure
Echo-Code
│
├── src/
│ ├── models/
│ │ ├── ef/
│ │ ├── quality/
│ │ └── segmentation/
│ │
│ └── inference/
│
├── scripts/
│ ├── train_echo_seg.py
│ ├── train_ef.py
│ ├── train_quality.py
│ └── cardiac_report.py
│
├── checkpoints/
├── data/
├── weights/
└── echo_codex_demo.ipynb


---

## Installation

Create a Python environment and install dependencies:
pip install -r requirements.txt


Core dependencies:
torch
torchvision
numpy
opencv-python
monai
albumentations
scikit-image
scikit-learn
matplotlib
tqdm


---

## Data

Due to GitHub storage limits, datasets and pretrained model checkpoints are hosted externally.

Google Drive:

https://drive.google.com/drive/folders/1mNYKmoGA0WOyQZ7atADTE1qTtkgYizRE

Download the folders:
datasets/
checkpoints/


Place them in the repository root:
Echo-Code/
├── data/
├── checkpoints/


Datasets used:

- EchoNet-Dynamic
- CAMUS
- CardiacUDA
- CACTUS
- PLOSONE

---

## Running the Pipeline

Example inference:
python scripts/cardiac_report.py --video demo.avi


Output includes:

- EF (regression)
- EF (segmentation derived)
- EDV / ESV
- Stroke Volume
- Cardiac Output
- Image quality score

---

## Training

Segmentation model:
python scripts/train_echo_seg.py

EF regression:
python scripts/train_ef.py

Quality model:
python scripts/train_quality.py


---

## Reproducing Results

Models were trained on Stanford FarmShare GPU nodes using PyTorch.

Evaluation metrics reported in the paper:

- Segmentation Dice: 0.860 (source), 0.784 (target)
- EF regression RMSE: 5.38
- Quality classification accuracy: 97.3%

---

## Paper

See the project paper:

Echo-Code_Paper.pdf

---

## License

Academic use only.

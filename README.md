@'
# AI-Based Camera Model and Individual Source Device Identification

An image source identification project combining Deep Learning and Photo Response Non-Uniformity (PRNU) analysis to identify the camera device associated with a digital image.

## Technologies
- Python
- PyTorch and Torchvision
- DeiT-Tiny (timm)
- NumPy and SciPy
- Pillow
- PyWavelets
- Streamlit

## Methodology

### 1. Deep Learning
A DeiT-Tiny Vision Transformer is trained to classify images across 22 camera-device classes.

### 2. PRNU Analysis
PRNU analysis extracts sensor-specific noise patterns using wavelet-based noise residual extraction and compares them against stored device fingerprints.

### 3. Score Fusion
The system combines deep-learning and PRNU matching scores to rank candidate camera devices.

Selected validation weights:
- DeiT-Tiny: 65%
- PRNU: 35%

## Reported Evaluation Results

Held-out test results from the current 22-device experiment:

| Metric | DeiT-Tiny | PRNU | Fusion |
|---|---:|---:|---:|
| Top-1 Accuracy | 80.60% | 87.40% | 96.20% |
| Top-3 Accuracy | 95.40% | 90.80% | 99.20% |
| Macro F1 | 77.30% | 85.24% | 95.84% |
| Weighted F1 | 80.16% | 85.79% | 96.13% |

These figures apply to the evaluated test split and do not establish performance on unseen camera devices.

## Project Structure

```text
camera-source/
├── experiments/
│   └── vision_deit_tiny/
│       ├── src/
│       └── prnu/
│           ├── src/
│           └── outputs/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
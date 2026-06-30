# MATE: Multimodal Air-quality Tracking at the Edge 🌍☁️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-Stable-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **MATE** (Multimodal Air-quality Tracking at the Edge), a Y-shaped dual-branch deep learning framework for real-time joint estimation of ambient PM2.5 and PM10 concentrations.

## 🌟 Overview

Traditional physical air quality monitoring stations are spatially sparse, creating observational blind spots. While image-based estimation offers a cost-effective alternative, unimodal vision networks are fundamentally confounded by non-pollution meteorological factors (e.g., high-humidity fog). 

**MATE** overcomes these limitations by utilizing a Y-shaped dual-branch topology:
- **Vision Branch (ResNet-18)**: Extracts deep spatial-visual descriptors from high-definition ambient outdoor imagery.
- **Tabular Branch (Attentive TabNet)**: Dynamically filters contextual noise from a 13-dimensional structured vector of environmental parameters (meteorological & geographical priors) using Gated Linear Units (GLU).
- **Y-Shaped Multi-task Head**: Fuses decoupled embeddings and utilizes a self-adaptive multi-task learning head governed by homoscedastic uncertainty weighting to balance the joint optimization of PM2.5 and PM10.

## 🚀 Hardware & Edge Deployment
The MATE framework is exceptionally lightweight and has been empirically validated for continuous in-situ field deployment on edge-computing hardware:
- **Computing Node**: NVIDIA Jetson Orin Nano
- **Sensors**: A8 mini camera (Vision) & SDS011 sensor (Ground-truth PM)
- **Performance**: ~14.5 FPS inference throughput | 12.0% CPU usage | Safe thermal profile (51.2°C).

## 📂 Repository Structure

```text
├── data/
│   ├── images/                  # Ambient outdoor imagery
│   └── tabular/                 # Environmental tabular data (Format: Algorithm-ID-Date.csv)
├── code/
│   ├── models/
│   │   ├── model_3poll.py       # Core implementation of the MATE Y-shaped architecture
│   │   └── layers.py            # Custom layers (GLU, AttentiveFeatureSelection)
│   ├── train.py                 # Script for end-to-end model training
│   └── explain_model.py         # XAI script (TabNet Feature Importance, Grad-CAM, SHAP)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
Installation
Clone this repository and install the required dependencies:
git clone [https://github.com/Maohsiuhsu/MATE-Air-Quality.git](https://github.com/Maohsiuhsu/MATE-Air-Quality.git)
cd MATE-Air-Quality
pip install -r requirements.txt

Quick Start
1. Training the Model
To train the MATE framework from scratch using the default configuration:

Bash
python code/train.py --batch_size 32 --epochs 100 --ablation_mode both







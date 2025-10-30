
# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication

Official PyTorch implementation of the paper:

**[Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication](https://arxiv.org/abs/2509.09597)**  
*Maysam Behmanesh, Erkan Turan, Maks Ovsjanikov*

## Overview
This repository contains implementations for **graph alignment** and **vision–language alignment** tasks introduced in the paper.

### Directory Structure
```
GADL-graph/     # Graph alignment tasks
GADL-VL/        # Vision–language alignment tasks
data/           # Datasets and precomputed embeddings
```

### Data Download
The data used in our experiments can be downloaded from:

https://mega.nz/folder/53YWVJLL#nnBta-z8z4UQzeSQaAb80A

Extract the contents into the `data/` directory before running experiments.

## Vision–Language Data Structure
The `data/vision-language/` folder contains precomputed embeddings for **CIFAR-10**, **CINIC-10**, **CIFAR-100**, and **ImageNet-100** datasets, as well as corresponding language embeddings.

```
data/vision-language/
├── CIFAR-10/
├── CINIC-10/
├── CIFAR-100/
├── ImageNet-100/
├── Language/
│   └── prompt_5/                # Language embeddings for CIFAR-10 and CINIC-10
├── Language-100/
│   └── prompt_5/                # Language embeddings for CIFAR-100
└─ Language100-ImageNet/
    └── prompt_5/                # Language embeddings for ImageNet-100

```

## Installation
1. Clone the repository
```bash
# Clone the repository
git clone https://github.com/maysambehmanesh/GADL.git
```
2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## Running Experiments
### Vision–Language Alignment
To run alignment on the vision–language task, use:
```bash
python GADL-VL/alignment_VL.py
```

### Custom Training Configuration
All training configurations are managed through the Lightning CLI system.

You can modify hyperparameters, datasets, or model options in:
```bash
config.yml
```

## Citation
If you find this work useful, please cite:
```bash
@article{behmanesh2025gadl,
  title={Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication},
  author={Behmanesh, Maysam and Turan, Erkan and Ovsjanikov, Maks},
  journal={arXiv preprint arXiv:2509.09597},
  year={2025}
}
```

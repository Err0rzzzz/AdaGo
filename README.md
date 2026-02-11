# AdaGo
Will conduct some experiments on different deep learning tasks using AdaGO optimizer

The paper link is https://arxiv.org/pdf/2509.02981?
=======

Implementation / experiments for **AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates** (AdaGO).

> This repository contains training scripts, configs, and utilities to reproduce runs and plots used in our experiments.

---

## Dataset

This project uses CIFAR-100.

Download automatically via torchvision:

```python
from torchvision.datasets import CIFAR100

CIFAR100(root="./data", train=True, download=True)

## How to run 
>>> python train_cifar100_vit_adago.py --run_all --out_dir results --exp_name cifar100_vit --epochs 30 --batch_size 128
Will get file tree like:
results/
├── cifar100_vit_adago/
├── cifar100_vit_muon/
└── cifar100_vit_adamw_only/

# AdaGo
<<<<<<< HEAD
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

>>>>>>> c58e47e (Initial commit)

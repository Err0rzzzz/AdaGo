import os
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 强制无 GUI 后端，避免 tkinter 线程/析构崩溃
import matplotlib.pyplot as plt


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism can slow down; keep it reasonable for training.
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Orth operator via SVD
# Orth(M) = U V^T
# ----------------------------
@torch.no_grad()
def orth_svd(M: torch.Tensor) -> torch.Tensor:
    # M: (m, n), float32/float16 on GPU
    # Use full_matrices=False for speed; stable enough for our use.
    # torch.linalg.svd returns U, S, Vh
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


# ----------------------------
# Muon: Orthogonalized momentum, constant stepsize
# Update: p <- p - eta * min(||G||, gamma) * Orth(M)
# where M = mu*M + (1-mu)*G
# ----------------------------
class Muon(optim.Optimizer):
    def __init__(
        self,
        params,
        eta: float = 0.3,
        mu: float = 0.9,
        gamma: float = 1.0,
        eps: float = 1e-8,  # numerical safety
        weight_decay: float = 0.0,  # decoupled weight decay
    ):
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not (0.0 <= mu < 1.0):
            raise ValueError("mu must be in [0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        defaults = dict(eta=eta, mu=mu, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Dict[str, Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats = {"avg_alpha": 0.0, "num_tensors": 0}

        for group in self.param_groups:
            eta = group["eta"]
            mu = group["mu"]
            gamma = group["gamma"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse grads.")

                # We only expect 2D tensors here (Linear weight)
                if p.ndim != 2:
                    continue

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)

                m = state["m"]
                # Momentum update
                m.mul_(mu).add_(g, alpha=(1.0 - mu))

                # Clamp current grad norm (paper scales by min(||G||, gamma))
                g_norm = torch.linalg.norm(g).clamp_min(eps)
                clamp = min(g_norm.item(), gamma)

                # Orthogonalize momentum
                O = orth_svd(m)

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1.0 - eta * wd)

                # Parameter update
                p.add_(O, alpha=(-eta * clamp))

                stats["avg_alpha"] += eta
                stats["num_tensors"] += 1

        if stats["num_tensors"] > 0:
            stats["avg_alpha"] /= stats["num_tensors"]
        stats["loss"] = loss
        return stats


# ----------------------------
# AdaGO: Adaptive stepsize for orth updates
# v_sq <- v_sq + min(||G||^2, gamma^2)
# alpha <- max(eps_stepsize, eta * min(||G||, gamma) / sqrt(v_sq))
# update magnitude uses clamp = min(||G||, gamma)
# p <- p - alpha * clamp * Orth(M)
# ----------------------------
class AdaGO(optim.Optimizer):
    def __init__(
        self,
        params,
        eta: float = 0.3,
        mu: float = 0.9,
        gamma: float = 1.0,
        v0: float = 1.0,
        eps_stepsize: float = 1e-4,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not (0.0 <= mu < 1.0):
            raise ValueError("mu must be in [0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if v0 <= 0:
            raise ValueError("v0 must be > 0")
        if eps_stepsize <= 0:
            raise ValueError("eps_stepsize must be > 0")
        defaults = dict(
            eta=eta,
            mu=mu,
            gamma=gamma,
            v0=v0,
            eps_stepsize=eps_stepsize,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Dict[str, Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stats = {"avg_alpha": 0.0, "avg_clamp": 0.0, "num_tensors": 0}

        for group in self.param_groups:
            eta = group["eta"]
            mu = group["mu"]
            gamma = group["gamma"]
            v0 = group["v0"]
            eps_stepsize = group["eps_stepsize"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdaGO does not support sparse grads.")
                if p.ndim != 2:
                    continue

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v_sq" not in state:
                    # v0 is initialized as v0 > 0; we store v_sq
                    state["v_sq"] = torch.tensor(float(v0 * v0), device=p.device, dtype=torch.float32)

                m = state["m"]
                v_sq = state["v_sq"]

                # Momentum update
                m.mul_(mu).add_(g, alpha=(1.0 - mu))

                # Current grad norm
                g_norm = torch.linalg.norm(g).clamp_min(eps)
                g_norm_val = g_norm.item()
                clamp = min(g_norm_val, gamma)

                # Accumulator update: v_sq += min(||G||^2, gamma^2)
                inc = min(g_norm_val * g_norm_val, gamma * gamma)
                v_sq.add_(inc)

                v = torch.sqrt(v_sq).item()
                alpha = max(eps_stepsize, eta * clamp / max(v, eps))

                # Orthogonalize momentum
                O = orth_svd(m)

                # Decoupled weight decay (using alpha as stepsize proxy)
                if wd > 0:
                    p.mul_(1.0 - alpha * wd)

                # Update: p <- p - alpha * clamp * O
                p.add_(O, alpha=(-alpha * clamp))

                stats["avg_alpha"] += alpha
                stats["avg_clamp"] += clamp
                stats["num_tensors"] += 1

        if stats["num_tensors"] > 0:
            stats["avg_alpha"] /= stats["num_tensors"]
            stats["avg_clamp"] /= stats["num_tensors"]
        stats["loss"] = loss
        return stats


# ----------------------------
# Utils: accuracy
# ----------------------------
@torch.no_grad()
def top1_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ----------------------------
# Build dataset/dataloaders
# ----------------------------
def build_cifar100_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    # ImageNet-style normalization (common when using ImageNet-pretrained weights)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False
    )
    return train_loader, test_loader


# ----------------------------
# Model: torchvision ViT-B/16 pretrained
# ----------------------------
def build_vit_b16(num_classes: int = 100) -> nn.Module:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    # Replace classifier head
    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)
    return model


# ----------------------------
# Param grouping: Linear weights -> AdaGO/Muon, others -> AdamW
# ----------------------------
def split_params_for_hybrid(model: nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    orth_params: List[torch.Tensor] = []
    other_params: List[torch.Tensor] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Use Orth optimizer only for 2D Linear weights
        if p.ndim == 2 and (".weight" in name) and ("heads.head" not in name or True):
            # include all linear weights (including head), still fine
            orth_params.append(p)
        else:
            other_params.append(p)

    return orth_params, other_params


# ----------------------------
# Train/eval loops
# ----------------------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    orth_avg_alpha: float
    orth_avg_clamp: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    opt_other: optim.Optimizer,
    opt_orth: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    alpha_sum = 0.0
    clamp_sum = 0.0
    alpha_n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt_other.zero_grad(set_to_none=True)
        opt_orth.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        # Important: step with scaler for both optimizers
        scaler.step(opt_other)
        orth_stats = scaler.step(opt_orth)  # returns None; so we manually call step later if needed

        # GradScaler doesn't forward return values; call step() under no scaling for stats
        # We can compute stats by calling opt_orth.step() AFTER unscale (but we'd double-update).
        # So: compute stats inside opt_orth.step() normally without scaler return is fine.
        # Workaround: unscale then call opt_orth.step() ourselves and skip scaler.step(opt_orth).
        # We'll do the correct method below: use scaler.unscale_ + manual step.
        #
        # Therefore, above scaler.step(opt_orth) is not used. We'll correct by re-implementing:
        #
        # --- Correction: ---
        # This block exists because many people copy patterns; we keep the correct implementation below.
        raise RuntimeError("Internal guard: use the correct AMP optimizer stepping (see below).")

    # unreachable
    return 0.0, 0.0, 0.0, 0.0


def train_one_epoch_amp_correct(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    opt_other: optim.Optimizer,
    opt_orth: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    alpha_sum = 0.0
    clamp_sum = 0.0
    alpha_n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt_other.zero_grad(set_to_none=True)
        opt_orth.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        # Optional: clip (after unscale)
        scaler.unscale_(opt_other)
        scaler.unscale_(opt_orth)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(opt_other)
        orth_stats = opt_orth.step()  # returns stats dict
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += top1_acc(logits, y) * bs
        n += bs

        if orth_stats and orth_stats.get("num_tensors", 0) > 0:
            alpha_sum += orth_stats.get("avg_alpha", 0.0)
            clamp_sum += orth_stats.get("avg_clamp", 0.0)
            alpha_n += 1

        pbar.set_postfix(loss=loss.item(), acc=total_acc / max(n, 1))

    avg_loss = total_loss / max(n, 1)
    avg_acc = total_acc / max(n, 1)
    avg_alpha = alpha_sum / max(alpha_n, 1)
    avg_clamp = clamp_sum / max(alpha_n, 1)
    return avg_loss, avg_acc, avg_alpha, avg_clamp


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc="eval", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += top1_acc(logits, y) * bs
        n += bs

        pbar.set_postfix(loss=loss.item(), acc=total_acc / max(n, 1))

    return total_loss / max(n, 1), total_acc / max(n, 1)


def plot_curves(logs: List[EpochLog], out_dir: str) -> None:
    epochs = [l.epoch for l in logs]
    train_loss = [l.train_loss for l in logs]
    test_loss = [l.test_loss for l in logs]
    train_acc = [l.train_acc for l in logs]
    test_acc = [l.test_acc for l in logs]
    alpha = [l.orth_avg_alpha for l in logs]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("top1_acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, alpha, label="orth_avg_alpha")
    plt.xlabel("epoch")
    plt.ylabel("avg stepsize (alpha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adago_alpha_curve.png"), dpi=200)
    plt.close()


def save_csv(logs: List[EpochLog], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,orth_avg_alpha,orth_avg_clamp\n")
        for l in logs:
            f.write(
                f"{l.epoch},{l.train_loss:.6f},{l.train_acc:.6f},{l.test_loss:.6f},{l.test_acc:.6f},"
                f"{l.orth_avg_alpha:.6f},{l.orth_avg_clamp:.6f}\n"
            )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./results_cifar100_vit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--optimizer", type=str, default="adago", choices=["adago", "muon", "adamw_only"])
    parser.add_argument("--eta", type=float, default=0.3)          # for AdaGO/Muon orth group
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--eps_stepsize", type=float, default=1e-4)
    parser.add_argument("--orth_wd", type=float, default=0.0)

    parser.add_argument("--adamw_lr", type=float, default=3e-4)    # for non-orth group
    parser.add_argument("--adamw_wd", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--run_all", action="store_true", help="Run adago/muon/adamw_only sequentially with separate folders.")
    parser.add_argument("--exp_name", type=str, default="cifar100_vit", help="Experiment prefix for folder naming.")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def run_once(run_optimizer: str, base_args: argparse.Namespace) -> None:
        # ---- create unique output dir for this run ----
        run_dir = os.path.join(base_args.out_dir, f"{base_args.exp_name}_{run_optimizer}")
        os.makedirs(run_dir, exist_ok=True)

        # ---- build loaders (data download happens here once; cached thereafter) ----
        train_loader, test_loader = build_cifar100_loaders(
            data_dir=base_args.data_dir,
            batch_size=base_args.batch_size,
            num_workers=base_args.num_workers,
            img_size=base_args.img_size,
        )

        # ---- build model fresh for each optimizer (fair comparison) ----
        model = build_vit_b16(num_classes=100).to(device)
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            model = nn.DataParallel(model)

        orth_params, other_params = split_params_for_hybrid(model)

        # ---- optimizers ----
        opt_other = optim.AdamW(other_params, lr=base_args.adamw_lr, weight_decay=base_args.adamw_wd)

        if run_optimizer == "adago":
            opt_orth = AdaGO(
                orth_params,
                eta=base_args.eta,
                mu=base_args.mu,
                gamma=base_args.gamma,
                v0=base_args.v0,
                eps_stepsize=base_args.eps_stepsize,
                weight_decay=base_args.orth_wd,
            )
        elif run_optimizer == "muon":
            opt_orth = Muon(
                orth_params,
                eta=base_args.eta,
                mu=base_args.mu,
                gamma=base_args.gamma,
                weight_decay=base_args.orth_wd,
            )
        elif run_optimizer == "adamw_only":
            opt_orth = optim.SGD(orth_params, lr=0.0)  # dummy
        else:
            raise ValueError(f"Unknown optimizer: {run_optimizer}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_other, T_max=base_args.epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        logs: List[EpochLog] = []
        best_acc = -1.0

        for epoch in range(1, base_args.epochs + 1):
            t0 = time.time()

            if run_optimizer == "adamw_only":
                # AdamW on all params baseline
                opt_all = optim.AdamW(model.parameters(), lr=base_args.adamw_lr, weight_decay=base_args.adamw_wd)
                train_loss, train_acc, _, _ = train_one_epoch_amp_correct(
                    model, train_loader, criterion,
                    opt_all, optim.SGD([], lr=0.0),
                    device, scaler,
                    max_grad_norm=base_args.max_grad_norm
                )
                avg_alpha, avg_clamp = 0.0, 0.0
            else:
                train_loss, train_acc, avg_alpha, avg_clamp = train_one_epoch_amp_correct(
                    model, train_loader, criterion,
                    opt_other, opt_orth,
                    device, scaler,
                    max_grad_norm=base_args.max_grad_norm
                )

            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            logs.append(EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                orth_avg_alpha=avg_alpha,
                orth_avg_clamp=avg_clamp,
            ))

            # save best checkpoint per run
            if test_acc > best_acc:
                best_acc = test_acc
                ckpt_path = os.path.join(run_dir, "best.pt")
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "test_acc": test_acc, "args": vars(base_args),
                     "optimizer": run_optimizer},
                    ckpt_path
                )

            # save logs/plots per epoch
            save_csv(logs, os.path.join(run_dir, "log.csv"))
            plot_curves(logs, run_dir)

            dt = time.time() - t0
            print(
                f"[{run_optimizer}] [epoch {epoch:03d}/{base_args.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                f"orth_alpha={avg_alpha:.6f} time={dt:.1f}s"
            )

        print(f"[{run_optimizer}] Done. Best test_acc = {best_acc:.4f}. Results in: {run_dir}")

    # ---- dispatcher ----
    if args.run_all:
        # Make runs deterministic & comparable: reset seed before each run
        for opt_name in ["adago", "muon", "adamw_only"]:
            set_seed(args.seed)
            run_once(opt_name, args)
    else:
        # single run (still isolated folder per optimizer)
        set_seed(args.seed)
        run_once(args.optimizer, args)


if __name__ == "__main__":
    main()

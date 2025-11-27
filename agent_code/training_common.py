from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.face_embedder.vision_transformer import vit_base


def load_chimpufe_backbone(pretrained_weights: Optional[str], device: torch.device):
    """
    Build the ViT backbone used by ChimpUFE and optionally load EMA teacher weights.
    """
    input_size = 224
    model = vit_base(
        img_size=input_size,
        patch_size=14,
        init_values=1e-05,
        ffn_layer='mlp',
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    if pretrained_weights:
        state = torch.load(pretrained_weights, map_location="cpu", weights_only=False)['teacher']
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def default_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size) if img_size >= 224 else transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def default_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TwoCropTransform:
    """
    Apply the same augmentation chain twice, returning two correlated views.
    Suitable for SimCLR/BYOL-style contrastive training.
    """
    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def contrastive_train_transforms(img_size=224):
    # SimCLR-ish transformations; adjust strength as needed.
    blur_kernel = int(0.1 * img_size // 2 * 2 + 1)  # odd kernel, ~10% of resolution
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ViTBackboneWrapper(nn.Module):
    """
    Wraps the ViT to always return a (B, C) embedding.
    Tries, in order:
      - dict with 'x_norm_clstoken'
      - tensor with shape (B, N, C): use CLS token ([:, 0, :]) or mean
      - tensor (B, C): use as-is
    """
    def __init__(self, vit: nn.Module, use_cls_token: bool = True):
        super().__init__()
        self.vit = vit
        self.use_cls_token = use_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.vit(x)

        if isinstance(y, dict):
            if 'x_norm_clstoken' in y and y['x_norm_clstoken'] is not None:
                return y['x_norm_clstoken']
            if 'x_norm_patchtokens' in y and y['x_norm_patchtokens'] is not None:
                t = y['x_norm_patchtokens']
                return t[:, 0] if self.use_cls_token else t.mean(dim=1)

        if isinstance(y, torch.Tensor):
            if y.dim() == 3:
                return y[:, 0] if self.use_cls_token else y.mean(dim=1)
            if y.dim() == 2:
                return y
            raise RuntimeError(f"Unexpected tensor shape from ViT: {tuple(y.shape)}")

        raise RuntimeError("Unsupported output type from ViT backbone.")


def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


class ContrastiveImageFolder(Dataset):
    """
    Wrap ImageFolder to emit two augmented views and ignore labels.
    """
    def __init__(self, root: Path, transform: Callable):
        super().__init__()
        self.base = datasets.ImageFolder(str(root))
        self.base.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        (view1, view2), _ = self.base[idx]
        return view1, view2


def build_classification_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int,
                                     val_split: float = 0.1):
    tf_train = default_train_transforms(img_size)
    tf_val = default_val_transforms(img_size)
    full = datasets.ImageFolder(str(data_root), transform=tf_train)
    class_names = full.classes

    n_total = len(full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])
    val_set.dataset.transform = tf_val

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(class_names), class_names


def build_contrastive_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int,
                                  val_split: float = 0.1):
    tf_two_crop = TwoCropTransform(contrastive_train_transforms(img_size))
    tf_val = TwoCropTransform(default_val_transforms(img_size))
    full = ContrastiveImageFolder(data_root, transform=tf_two_crop)

    n_total = len(full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])
    # Re-wrap transforms for val to reduce stochasticity
    val_set.dataset.base.transform = tf_val

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, val_loader


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Symmetric NT-Xent (SimCLR) loss. Expects two batches of projected features.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    # Remove self-similarities
    diag_mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask, float("-inf"))

    pos = torch.cat([torch.diag(sim, z1.size(0)), torch.diag(sim, -z1.size(0))], dim=0)
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(pos) / denom).mean()
    return loss


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def unfreeze_last_blocks(vit: nn.Module, n_blocks: int):
    if n_blocks <= 0:
        return
    blocks = getattr(vit, "blocks", None)
    if blocks is not None and len(blocks) >= n_blocks:
        for b in blocks[-n_blocks:]:
            set_trainable(b, True)


def create_run_dir(out_root: Path, data_root: Path) -> Path:
    """
    Create a timestamped run directory under out_root using the dataset folder name.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = data_root.name or "run"
    run_dir = out_root / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

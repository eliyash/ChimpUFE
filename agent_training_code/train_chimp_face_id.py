from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# ------------------
# Utility helpers
# ------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chimp_face_id")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_dir / "training.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_embed_dim(backbone: nn.Module) -> int:
    for attr in ("num_features", "embed_dim"):
        if hasattr(backbone, attr):
            val = getattr(backbone, attr)
            if isinstance(val, int):
                return val
    raise AttributeError("Could not infer embedding dimension from backbone. Please specify manually.")


# ------------------
# Data
# ------------------


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class UnlabeledPairDataset(Dataset):
    """Unlabeled dataset that returns two augmented views per image."""

    def __init__(self, files: List[Path], transform: transforms.Compose):
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.transform(img)


class LabeledPairDataset(Dataset):
    """ImageFolder-style dataset returning two augmented views and a label."""

    def __init__(self, root: Path, transform_view1: transforms.Compose, transform_view2: Optional[transforms.Compose] = None):
        self.base = datasets.ImageFolder(root)
        self.transform1 = transform_view1
        self.transform2 = transform_view2 or transform_view1

    def __len__(self) -> int:
        return len(self.base.samples)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        v1 = self.transform1(img)
        v2 = self.transform2(img)
        return v1, v2, label


class EvalImageDataset(Dataset):
    """Simple eval dataset with a single view for inference/export."""

    def __init__(self, root: Path, transform: transforms.Compose):
        self.base = datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]

    @property
    def classes(self) -> List[str]:
        return self.base.classes


# ------------------
# Models and losses
# ------------------


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupConLoss(nn.Module):
    """Supervised contrastive loss from https://arxiv.org/abs/2004.11362."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        bsz, n_views = features.shape[:2]
        features = F.normalize(features, dim=2)
        features = features.view(bsz * n_views, -1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        loss = loss.view(bsz, n_views).mean()
        return loss


class ArcMarginProduct(nn.Module):
    """ArcFace head for classification with additive angular margin."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(reps, reps.T) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    logits = logits.masked_fill(mask, -9e15)
    targets = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(z1.device)
    loss = F.cross_entropy(logits, targets)
    return loss


# ------------------
# Backbone loader
# ------------------


def load_dinov3_backbone(model_name: str, device: torch.device, freeze: bool = False) -> nn.Module:
    """Loads a ViT backbone (DINOV3) using timm."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Please install timm: pip install timm (needed for DINOV3 backbones)") from exc

    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
    except Exception as exc:
        available = [m for m in timm.list_models("*dinov3*")]
        raise ValueError(
            f"Could not load backbone '{model_name}'. "
            f"Available DINOV3 models in this timm version: {available}"
        ) from exc
    model.to(device)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


# ------------------
# Transforms
# ------------------


def get_ssl_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_supervised_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ------------------
# Training: self-supervised
# ------------------


def build_ssl_loaders(data_root: Path, img_size: int, batch_size: int, workers: int, val_split: float) -> Tuple[DataLoader, Optional[DataLoader]]:
    files = [p for p in Path(data_root).rglob("**/*") if p.suffix.lower() in IMG_EXTS]
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images for contrastive learning, found {len(files)} under {data_root}")
    random.shuffle(files)
    val_size = int(len(files) * val_split)
    val_files = files[:val_size]
    train_files = files[val_size:] if val_size < len(files) else files

    transform = get_ssl_transform(img_size)
    train_ds = UnlabeledPairDataset(train_files, transform)
    val_ds = UnlabeledPairDataset(val_files, transform) if val_files else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader


class SSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_hidden: int, proj_out: int):
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(get_embed_dim(backbone), hidden_dim=proj_hidden, out_dim=proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.projection(feats)


def train_self_supervised(backbone: nn.Module, args: argparse.Namespace, logger: logging.Logger, writer: SummaryWriter, device: torch.device, run_dir: Path) -> nn.Module:
    train_loader, val_loader = build_ssl_loaders(
        Path(args.ssl_data),
        img_size=args.img_size,
        batch_size=args.ssl_batch_size,
        workers=args.workers,
        val_split=args.ssl_val_split,
    )

    model = SSLModel(backbone, proj_hidden=args.proj_hidden, proj_out=args.proj_out).to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.ssl_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ssl_epochs)

    best_val = float("inf")
    best_path = run_dir / "selfsup_best.pt"
    last_path = run_dir / "selfsup_last.pt"

    for epoch in range(1, args.ssl_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (v1, v2) in train_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature=args.ssl_temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * v1.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_loader.dataset)
        writer.add_scalar("ssl/train_loss", train_loss, epoch)
        writer.add_scalar("ssl/lr", scheduler.get_last_lr()[0], epoch)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for (v1, v2) in val_loader:
                    v1 = v1.to(device)
                    v2 = v2.to(device)
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = nt_xent_loss(z1, z2, temperature=args.ssl_temperature)
                    v_loss += loss.item() * v1.size(0)
            val_loss = v_loss / len(val_loader.dataset)
            writer.add_scalar("ssl/val_loss", val_loss, epoch)

        logger.info(f"[SSL] Epoch {epoch:03d}/{args.ssl_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, best_path)

    best_state = torch.load(best_path, map_location=device)["state_dict"]
    backbone_state = {k.replace("backbone.", ""): v for k, v in best_state.items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_state, strict=False)
    return backbone


# ------------------
# Training: supervised fine-tuning
# ------------------


def build_supervised_loaders(data_root: Path, img_size: int, batch_size: int, workers: int, val_split: float) -> Tuple[DataLoader, DataLoader, EvalImageDataset]:
    train_tf, eval_tf = get_supervised_transforms(img_size)
    train_pair_ds = LabeledPairDataset(data_root, transform_view1=train_tf, transform_view2=train_tf)
    val_ds_full = EvalImageDataset(data_root, transform=eval_tf)

    val_size = int(len(train_pair_ds) * val_split)
    if val_split > 0 and val_size == 0 and len(train_pair_ds) > 1:
        val_size = 1
    train_size = len(train_pair_ds) - val_size

    if val_size > 0:
        train_ds, val_indices = random_split(train_pair_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        val_subset = Subset(val_ds_full, val_indices.indices)
    else:
        train_ds = train_pair_ds
        val_subset = val_ds_full

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader, val_ds_full


class SupervisedModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, use_classification: bool, supcon_head: bool, proj_hidden: int, proj_out: int):
        super().__init__()
        self.backbone = backbone
        self.use_classification = use_classification
        self.supcon_head = supcon_head
        embed_dim = get_embed_dim(backbone)
        self.projection = ProjectionHead(embed_dim, hidden_dim=proj_hidden, out_dim=proj_out) if supcon_head else None
        self.arc_margin = ArcMarginProduct(embed_dim, num_classes) if use_classification else None

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        feats = self.backbone(x)
        feats = F.normalize(feats, dim=1)
        logits = self.arc_margin(feats, labels) if self.use_classification and labels is not None else None
        proj = self.projection(feats) if self.supcon_head else None
        return feats, logits, proj


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def evaluate(model: SupervisedModel, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=args.supcon_temperature)

    losses = []
    accs = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                feats, logits, _ = model(images, labels if args.train_classification else None)
                batch_losses = []
                if args.train_classification and logits is not None:
                    batch_losses.append(ce_loss_fn(logits, labels))
                    accs.append(compute_accuracy(logits, labels))
                loss = sum(batch_losses) if batch_losses else torch.tensor(0.0, device=device)
            else:
                v1, v2, labels = batch
                v1 = v1.to(device)
                v2 = v2.to(device)
                labels = labels.to(device)
                feats1, logits, proj1 = model(v1, labels if args.train_classification else None)
                feats2, _, proj2 = model(v2, labels if args.train_classification else None)
                losses_batch = []
                if args.train_classification and logits is not None:
                    losses_batch.append(ce_loss_fn(logits, labels))
                    accs.append(compute_accuracy(logits, labels))
                if args.use_supcon or not args.train_classification:
                    feats_stack = torch.stack([proj1 if proj1 is not None else feats1, proj2 if proj2 is not None else feats2], dim=1)
                    losses_batch.append(args.supcon_weight * supcon_loss_fn(feats_stack, labels))
                loss = sum(losses_batch) if losses_batch else torch.tensor(0.0, device=device)

            losses.append(loss.item())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
    }


def train_supervised(backbone: nn.Module, args: argparse.Namespace, logger: logging.Logger, writer: SummaryWriter, device: torch.device, run_dir: Path) -> Tuple[SupervisedModel, Dict[str, float], List[str]]:
    train_loader, val_loader, full_eval_ds = build_supervised_loaders(
        Path(args.sup_data),
        img_size=args.img_size,
        batch_size=args.sup_batch_size,
        workers=args.workers,
        val_split=args.sup_val_split,
    )
    num_classes = len(full_eval_ds.classes)
    model = SupervisedModel(
        backbone=backbone,
        num_classes=num_classes,
        use_classification=args.train_classification,
        supcon_head=args.use_supcon or not args.train_classification,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
    ).to(device)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.sup_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sup_epochs)
    ce_loss_fn = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=args.supcon_temperature)

    best_metric = -float("inf")
    best_path = run_dir / "supervised_best.pt"
    last_path = run_dir / "supervised_last.pt"
    history: Dict[str, float] = {}

    for epoch in range(1, args.sup_epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        for (v1, v2, labels) in train_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            feats1, logits, proj1 = model(v1, labels if args.train_classification else None)

            losses = []
            if args.train_classification and logits is not None:
                ce = ce_loss_fn(logits, labels)
                losses.append(ce)
                total_acc += compute_accuracy(logits, labels)

            if args.use_supcon or not args.train_classification:
                feats2, _, proj2 = model(v2, labels if args.train_classification else None)
                feats_stack = torch.stack([proj1 if proj1 is not None else feats1, proj2 if proj2 is not None else feats2], dim=1)
                con_loss = supcon_loss_fn(feats_stack, labels)
                losses.append(args.supcon_weight * con_loss)

            loss = sum(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        avg_acc = (total_acc / total_batches) if args.train_classification else float("nan")
        writer.add_scalar("sup/train_loss", avg_loss, epoch)
        if args.train_classification:
            writer.add_scalar("sup/train_acc", avg_acc, epoch)
        writer.add_scalar("sup/lr", scheduler.get_last_lr()[0], epoch)

        val_metrics = evaluate(model, val_loader, device, args)
        writer.add_scalar("sup/val_loss", val_metrics["loss"], epoch)
        if args.train_classification:
            writer.add_scalar("sup/val_acc", val_metrics["acc"], epoch)

        logger.info(
            f"[SUP] Epoch {epoch:03d}/{args.sup_epochs} "
            f"train_loss={avg_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc={avg_acc:.4f} val_acc={val_metrics.get('acc', float('nan')):.4f}"
        )

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, last_path)
        score = val_metrics["acc"] if args.train_classification else -val_metrics["loss"]
        if score > best_metric:
            best_metric = score
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, best_path)
            history.update({"best_epoch": epoch, "best_val_acc": val_metrics.get("acc", float("nan")), "best_val_loss": val_metrics["loss"]})

    best_state = torch.load(best_path, map_location=device)["state_dict"]
    model.load_state_dict(best_state, strict=True)
    return model, history, full_eval_ds.classes


# ------------------
# Export: embeddings and confusion matrix
# ------------------


def export_embeddings(model: SupervisedModel, dataset: EvalImageDataset, device: torch.device, out_path: Path) -> None:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats, _, _ = model(images, targets.to(device) if model.use_classification else None)
            embeddings.append(feats.cpu())
            labels.append(targets)
    embeddings_tensor = torch.cat(embeddings, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    np.savez(out_path, embeddings=embeddings_tensor.numpy(), labels=labels_tensor.numpy(), classes=np.array(dataset.classes))


def compute_confusion_matrix(model: SupervisedModel, dataset: EvalImageDataset, device: torch.device, out_path: Path) -> None:
    if not model.use_classification:
        return
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("scikit-learn and matplotlib are required for confusion matrix export.") from exc

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            _, logits, _ = model(images, targets)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    cm = confusion_matrix(labels_tensor.numpy(), preds_tensor.numpy(), labels=list(range(len(dataset.classes))))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(dataset.classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(dataset.classes, rotation=45, ha="right")
    ax.set_yticklabels(dataset.classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------
# Argument parsing
# ------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chimp face identification training (self-supervised + supervised) using DINOV3 backbone.")
    parser.add_argument("--ssl_data", required=True, help="Path to unlabeled faces folder (images only).")
    parser.add_argument("--sup_data", required=True, help="Path to labeled faces folder (ImageFolder: class subfolders = identities).")
    parser.add_argument("--output_dir", required=True, help="Where to save logs, checkpoints, embeddings.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--train_classification", default=True, action=argparse.BooleanOptionalAction, help="Enable classification head fine-tuning (ArcFace). Use --no-train-classification to disable.")

    # self-supervised
    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--ssl_batch_size", type=int, default=64)
    parser.add_argument("--ssl_lr", type=float, default=3e-4)
    parser.add_argument("--ssl_temperature", type=float, default=0.2)
    parser.add_argument("--ssl_val_split", type=float, default=0.1)

    # supervised
    parser.add_argument("--sup_epochs", type=int, default=30)
    parser.add_argument("--sup_batch_size", type=int, default=32)
    parser.add_argument("--sup_lr", type=float, default=1e-4)
    parser.add_argument("--sup_val_split", type=float, default=0.1)
    parser.add_argument("--use_supcon", action="store_true", help="Add supervised contrastive loss on top of classification.")
    parser.add_argument("--supcon_weight", type=float, default=1.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.1)

    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_base_patch16_dinov3",  # DINOV3 ViT base (adjust to any entry from timm.list_models('*dinov3*'))
        help="timm model name for DINOV3 backbone (see timm.list_models('*dinov3*')).",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--proj_hidden", type=int, default=2048)
    parser.add_argument("--proj_out", type=int, default=256)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


# ------------------
# Main
# ------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(run_dir)
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")
    save_config(args, run_dir)

    device = get_device(force_cpu=args.cpu)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading backbone: {args.backbone}")
    backbone = load_dinov3_backbone(args.backbone, device=device, freeze=args.freeze_backbone)

    logger.info("Starting self-supervised phase...")
    backbone = train_self_supervised(backbone, args, logger, writer, device, run_dir)
    logger.info("Self-supervised phase completed.")

    logger.info("Starting supervised fine-tuning phase...")
    model, history, class_names = train_supervised(backbone, args, logger, writer, device, run_dir)
    logger.info(f"Supervised phase completed. Best metrics: {history}")

    embed_path = run_dir / "embeddings.npz"
    logger.info(f"Exporting embeddings to {embed_path}")
    full_eval_ds = EvalImageDataset(Path(args.sup_data), transform=get_supervised_transforms(args.img_size)[1])
    export_embeddings(model, full_eval_ds, device, embed_path)

    if args.train_classification:
        cm_path = run_dir / "confusion_matrix.png"
        try:
            compute_confusion_matrix(model, full_eval_ds, device, cm_path)
            cm_img = torch.from_numpy(np.array(Image.open(cm_path).convert("RGB"))).permute(2, 0, 1)
            writer.add_image("confusion_matrix", cm_img, dataformats="CHW", global_step=0)
            logger.info(f"Confusion matrix saved to {cm_path}")
        except ImportError as exc:
            logger.warning(f"Could not generate confusion matrix (missing dependency): {exc}")

    writer.flush()
    writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()

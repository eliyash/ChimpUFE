# train_chimpufe_finetune.py
from __future__ import annotations
import argparse, math, os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.face_embedder.vision_transformer import vit_base


# ---- You already have these two in your project ----
# from dinov2.models.vision_transformer import vit_base  # or your local vit_base definition
# ----------------------------------------------------

# ---------- Backbone loading (as you provided) ----------
def load_chimpufe_backbone(pretrained_weights: str, device: torch.device):
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
    # Load EMA teacher checkpoint (as per ChimpUFE inference)
    state = torch.load(pretrained_weights, map_location="cpu", weights_only=False)['teacher']
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model

def default_train_transforms(img_size=224):
    # Strong-ish but safe for faces; tune if needed
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

# ---------- Robust feature extraction ----------
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

        # Case 1: dict output (common in DINOv2)
        if isinstance(y, dict):
            if 'x_norm_clstoken' in y and y['x_norm_clstoken'] is not None:
                return y['x_norm_clstoken']  # (B, C)
            if 'x_norm_patchtokens' in y and y['x_norm_patchtokens'] is not None:
                t = y['x_norm_patchtokens']  # (B, N, C)
                return t[:, 0] if self.use_cls_token else t.mean(dim=1)

        # Case 2: tensor
        if isinstance(y, torch.Tensor):
            if y.dim() == 3:  # (B, N, C)
                return y[:, 0] if self.use_cls_token else y.mean(dim=1)
            if y.dim() == 2:  # (B, C)
                return y
            raise RuntimeError(f"Unexpected tensor shape from ViT: {tuple(y.shape)}")

        raise RuntimeError("Unsupported output type from ViT backbone.")

# ---------- Heads ----------
class IdentityHead(nn.Module):
    """Linear classifier for softmax cross-entropy."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # logits

class ArcMarginProduct(nn.Module):
    """
    ArcFace margin head.
    Ref: ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Normalize features and weights
        x = F.normalize(embeddings)
        W = F.normalize(self.weight)
        # Cosine logits
        cos = F.linear(x, W)  # (B, C)
        sin = torch.sqrt(torch.clamp(1.0 - cos**2, min=0.0))
        cos_m = cos * self.cos_m - sin * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            cond = (cos > 0).float()
            adjusted = cond * cos_m + (1 - cond) * cos
        else:
            cond = (cos > self.th).float()
            adjusted = cond * cos_m - (1 - cond) * (cos - self.mm)

        # One-hot scatter margin
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * adjusted + (1.0 - one_hot) * cos)
        return output  # logits to feed into CrossEntropyLoss

# ---------- Model wrapper ----------
class ChimpUFEFinetune(nn.Module):
    def __init__(self, vit_backbone: nn.Module, embed_dim: int, head_type: str, num_classes: int,
                 arcface_s: float = 30.0, arcface_m: float = 0.5):
        super().__init__()
        self.backbone = ViTBackboneWrapper(vit_backbone, use_cls_token=True)
        self.head_type = head_type
        if head_type == "softmax":
            self.head = IdentityHead(embed_dim, num_classes)
        elif head_type == "arcface":
            self.head = ArcMarginProduct(embed_dim, num_classes, s=arcface_s, m=arcface_m)
        else:
            raise ValueError("head_type must be 'softmax' or 'arcface'.")

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        z = self.backbone(x)  # (B, C)
        if self.head_type == "softmax":
            logits = self.head(z)
            return {"emb": z, "logits": logits}
        else:
            if labels is None:
                raise ValueError("labels required for arcface forward")
            logits = self.head(z, labels)
            return {"emb": z, "logits": logits}

# ---------- Training utilities ----------
def build_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int,
                      val_split: float = 0.1):
    tf_train = default_train_transforms(img_size)
    tf_val = default_val_transforms(img_size)
    full = datasets.ImageFolder(str(data_root), transform=tf_train)
    class_names = full.classes

    # Simple split by index
    n_total = len(full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])
    # Use val transforms for val subset
    val_set.dataset.transform = tf_val

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(class_names), class_names

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def set_trainable(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def make_optimizers(model: ChimpUFEFinetune, lr_backbone: float, lr_head: float, weight_decay: float):
    return torch.optim.AdamW([
        {"params": (p for p in model.backbone.parameters() if p.requires_grad), "lr": lr_backbone},
        {"params": (p for p in model.head.parameters() if p.requires_grad), "lr": lr_head},
    ], weight_decay=weight_decay)

# ---------- Main train loop ----------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    vit = load_chimpufe_backbone(args.weights, device)
    # Optionally freeze the backbone (or partially)
    if args.freeze_backbone:
        set_trainable(vit, False)

    # Heuristic: DINOv2 ViT-B default embed dim is 768
    embed_dim = getattr(vit, "embed_dim", 768)

    model = ChimpUFEFinetune(
        vit_backbone=vit,
        embed_dim=embed_dim,
        head_type=args.head,
        num_classes=args.num_classes or 0,  # replaced after loaders if 0
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
    ).to(device)

    train_loader, val_loader, n_classes, class_names = build_dataloaders(
        Path(args.data_root), img_size=224, batch_size=args.batch_size,
        num_workers=args.workers, val_split=args.val_split
    )

    # If user didn’t specify num_classes, set from data
    if args.num_classes == 0:
        if isinstance(model.head, IdentityHead):
            model.head = IdentityHead(embed_dim, n_classes).to(device)
        else:
            model.head = ArcMarginProduct(embed_dim, n_classes, s=args.arcface_s, m=args.arcface_m).to(device)

    # Unfreeze last blocks if requested
    if args.unfreeze_last_n_blocks > 0:
        # Try to find transformer blocks and unfreeze last N
        blocks = getattr(vit, "blocks", None)
        if blocks is not None and len(blocks) >= args.unfreeze_last_n_blocks:
            for b in blocks[-args.unfreeze_last_n_blocks:]:
                set_trainable(b, True)

    optimizer = make_optimizers(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss (both heads produce logits)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(images, labels if args.head == "arcface" else None)
            logits = out["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_top1(logits, labels)
            train_loss += loss.item() * images.size(0)
            train_acc  += acc * images.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc  /= len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images, labels if args.head == "arcface" else None)
                logits = out["logits"]
                loss = criterion(logits, labels)
                acc = accuracy_top1(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_acc  += acc * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
              f"lr={[g['lr'] for g in optimizer.param_groups]}")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "best_val_acc": best_val,
            }
            torch.save(ckpt, outdir / "best.ckpt")

    # Final save
    torch.save(model.state_dict(), outdir / "last_state_dict.pt")
    print(f"Done. Best val acc: {best_val:.3f}. Saved to: {outdir}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Finetune ChimpUFE ViT-B/14 backbone")
    p.add_argument("--weights", type=str, required=True, help="Path to ChimpUFE checkpoint (.pt/.pth with 'teacher')")
    p.add_argument("--data_root", type=str, required=True, help="ImageFolder root with class subfolders")
    p.add_argument("--out_dir", type=str, default="./finetune_outputs")

    p.add_argument("--head", type=str, default="arcface", choices=["softmax", "arcface"])
    p.add_argument("--num_classes", type=int, default=0, help="If 0, inferred from data_root")
    p.add_argument("--arcface_s", type=float, default=30.0)
    p.add_argument("--arcface_m", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--freeze_backbone", action="store_true", help="Freeze entire ViT backbone")
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=4, help="Grad-enable last N transformer blocks")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

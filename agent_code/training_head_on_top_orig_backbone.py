# train_chimpufe_finetune.py
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training_common import (
    ViTBackboneWrapper,
    accuracy_top1,
    build_classification_dataloaders,
    create_run_dir,
    load_chimpufe_backbone,
    set_trainable,
    unfreeze_last_blocks,
)

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

    train_loader, val_loader, n_classes, class_names = build_classification_dataloaders(
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
        unfreeze_last_blocks(vit, args.unfreeze_last_n_blocks)

    optimizer = make_optimizers(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss (both heads produce logits)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    run_dir = create_run_dir(Path(args.out_dir), Path(args.data_root))
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(str(args.__dict__))

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

        log(f"[{epoch:03d}/{args.epochs}] "
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
            torch.save(ckpt, run_dir / "best.ckpt")

    # Final save
    torch.save(model.state_dict(), run_dir / "last_state_dict.pt")
    log(f"Done. Best val acc: {best_val:.3f}. Saved to: {run_dir}")

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

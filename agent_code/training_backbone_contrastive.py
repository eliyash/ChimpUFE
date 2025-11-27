from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_common import (
    ViTBackboneWrapper,
    build_contrastive_dataloaders,
    create_run_dir,
    load_chimpufe_backbone,
    nt_xent_loss,
    ProjectionHead,
    set_trainable,
    unfreeze_last_blocks,
)


class BackboneContrastive(nn.Module):
    """
    Backbone with projection head for self-supervised contrastive training.
    Returns projected vectors suitable for NT-Xent / SimCLR loss.
    """
    def __init__(self, vit_backbone: nn.Module, embed_dim: int, proj_hidden: int = 2048, proj_out: int = 256):
        super().__init__()
        self.backbone = ViTBackboneWrapper(vit_backbone, use_cls_token=True)
        self.proj = ProjectionHead(embed_dim, hidden_dim=proj_hidden, out_dim=proj_out)

    def forward(self, x):
        z = self.backbone(x)
        return self.proj(z)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    vit = load_chimpufe_backbone(args.weights, device)
    if args.freeze_backbone:
        set_trainable(vit, False)
    if args.unfreeze_last_n_blocks > 0:
        unfreeze_last_blocks(vit, args.unfreeze_last_n_blocks)

    embed_dim = getattr(vit, "embed_dim", 768)
    model = BackboneContrastive(
        vit_backbone=vit,
        embed_dim=embed_dim,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
    ).to(device)

    train_loader, val_loader = build_contrastive_dataloaders(
        Path(args.data_root), img_size=224, batch_size=args.batch_size,
        num_workers=args.workers, val_split=args.val_split
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    run_dir = create_run_dir(Path(args.out_dir), Path(args.data_root))
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(str(args.__dict__))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for view1, view2 in train_loader:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * view1.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for view1, view2 in val_loader:
                view1, view2 = view1.to(device), view2.to(device)
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)
                val_loss += loss.item() * view1.size(0)
        val_loss /= len(val_loader.dataset)

        log(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_loss": best_val,
            }
            torch.save(ckpt, run_dir / "best_contrastive.ckpt")

    torch.save(model.state_dict(), run_dir / "last_contrastive_state_dict.pt")
    log(f"Done. Best val loss: {best_val:.4f}. Saved to: {run_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Self-supervised contrastive training of ChimpUFE backbone (SimCLR-style)")
    p.add_argument("--weights", type=str, default=None, help="Path to ChimpUFE checkpoint (.pt/.pth with 'teacher'); optional")
    p.add_argument("--data_root", type=str, required=True, help="ImageFolder root with class subfolders")
    p.add_argument("--out_dir", type=str, default="./contrastive_outputs")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--freeze_backbone", action="store_true", help="Freeze entire ViT backbone")
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=4, help="Grad-enable last N transformer blocks")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_out", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

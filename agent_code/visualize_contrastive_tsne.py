from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from training_backbone_contrastive import BackboneContrastive
from training_common import (
    ViTBackboneWrapper,
    create_run_dir,
    default_val_transforms,
    load_chimpufe_backbone,
)


def build_splits(data_root: Path, val_split: float, img_size: int, seed: int = 0):
    tf = default_val_transforms(img_size)
    full = datasets.ImageFolder(str(data_root), transform=tf)
    n_total = len(full)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=gen)
    return train_set, val_set, full.classes


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    for images, lbls in loader:
        images = images.to(device, non_blocking=True)
        z = model.backbone(images)
        feats.append(z.cpu())
        labels.append(lbls)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def plot_tsne(
    emb_train: torch.Tensor,
    lbl_train: torch.Tensor,
    emb_val: torch.Tensor,
    lbl_val: torch.Tensor,
    class_names,
    out_path: Path,
):
    all_emb = torch.cat([emb_train, emb_val], dim=0).numpy()
    perp = max(5, min(30, len(all_emb) - 1))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perp)
    coords = tsne.fit_transform(all_emb)

    n_train = emb_train.size(0)
    coords_train = coords[:n_train]
    coords_val = coords[n_train:]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Use consistent colors for classes across splits
    cmap = plt.get_cmap("tab20")
    norm = plt.Normalize(vmin=0, vmax=max(lbl_train.max().item(), lbl_val.max().item()) + 0.999)

    sc_train = ax.scatter(
        coords_train[:, 0], coords_train[:, 1],
        c=lbl_train.numpy(), cmap=cmap, norm=norm,
        marker="x", alpha=0.8, label="train"
    )
    sc_val = ax.scatter(
        coords_val[:, 0], coords_val[:, 1],
        c=lbl_val.numpy(), cmap=cmap, norm=norm,
        marker="o", alpha=0.6, edgecolors="k", linewidths=0.3, label="val"
    )

    cbar = fig.colorbar(sc_train, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of backbone embeddings (X=train, O=val)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="t-SNE visualization for contrastive backbone embeddings")
    p.add_argument("--ckpt", type=str, required=True, help="Path to contrastive checkpoint (best_contrastive.ckpt)")
    p.add_argument("--data_root", type=str, required=True, help="ImageFolder root used during training")
    p.add_argument("--out_dir", type=str, default="./viz_outputs", help="Folder to save the plot")
    p.add_argument("--out_name", type=str, default="tsne.png")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1, help="Should match training split for train/val markers")
    p.add_argument("--seed", type=int, default=0, help="Deterministic split for viz")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    vit = load_chimpufe_backbone(pretrained_weights=None, device=device)
    embed_dim = getattr(vit, "embed_dim", 768)
    proj_hidden = ckpt_args.get("proj_hidden", 2048)
    proj_out = ckpt_args.get("proj_out", 256)
    model = BackboneContrastive(vit, embed_dim=embed_dim, proj_hidden=proj_hidden, proj_out=proj_out).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    train_set, val_set, class_names = build_splits(Path(args.data_root), args.val_split, img_size=224, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    emb_train, lbl_train = extract_embeddings(model, train_loader, device)
    emb_val, lbl_val = extract_embeddings(model, val_loader, device)

    run_dir = create_run_dir(Path(args.out_dir), Path(args.data_root))
    out_path = run_dir / args.out_name
    plot_tsne(emb_train, lbl_train, emb_val, lbl_val, class_names, out_path)
    print(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    main()

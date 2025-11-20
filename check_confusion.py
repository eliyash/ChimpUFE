import json, random, time
from pathlib import Path
from collections import defaultdict, OrderedDict
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from demo_face_rec import get_model, get_embedding
import matplotlib.pyplot as plt
import numpy as np

def tsne(tsne_path, Q, G, class_names):
    all_embs, all_labels, all_split = [], [], []  # split: 'Q' or 'G'
    for ci in class_names:
        # queries
        for e in Q[ci]:
            all_embs.append(e.cpu().numpy())
            all_labels.append(ci)
            all_split.append('Q')
        # galleries
        for e in G[ci]:
            all_embs.append(e.cpu().numpy())
            all_labels.append(ci)
            all_split.append('G')

    X = np.stack(all_embs, axis=0)  # (2*x*k, D)

    # Choose a sensible perplexity: ~ N/3 but clamped to [5, 50]
    N = X.shape[0]
    perp = int(max(5, min(50, N // 3)))

    tsne = TSNE(
        n_components=2,
        init="pca",
        # learning_rate="auto",
        learning_rate=500,
        perplexity=perp,
        random_state=42,
        # max_iter=1000,
        max_iter=2000,
        metric="cosine",
        # metric="euclidean",
        # early_exaggeration = 12
        early_exaggeration = 20
    )

    X2 = tsne.fit_transform(X)  # (N, 2)


    base = Path(tsne_path)
    out_dir = base.parent

    labels_np = np.array(all_labels)
    split_np  = np.array(all_split)

    def _slugify(s: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in s)[:80]

    per_class_out_folder = out_dir / 'per_class'
    per_class_out_folder.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        idx_target = (labels_np == cls)
        idx_q = idx_target & (split_np == 'Q')
        idx_g = idx_target & (split_np == 'G')
        idx_other = ~idx_target

        plt.figure(figsize=(9, 7))
        # others (blue, faint)
        plt.scatter(X2[idx_other, 0], X2[idx_other, 1],
                    c='blue', s=14, alpha=0.35, marker='o', linewidths=0)
        # this class: queries (green)
        plt.scatter(X2[idx_q, 0], X2[idx_q, 1],
                    c='green', s=32, alpha=0.95, marker='o', linewidths=0.2)
        # this class: galleries (yellow)
        plt.scatter(X2[idx_g, 0], X2[idx_g, 1],
                    c='yellow', s=32, alpha=0.95, marker='o', linewidths=0.2)

        # simple legend
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='green',
                   markeredgecolor='black', markersize=7, label=f'{cls} — Query'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='yellow',
                   markeredgecolor='black', markersize=7, label=f'{cls} — Gallery'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
                   markeredgecolor='none', markersize=7, alpha=0.35, label='Others'),
        ]
        plt.legend(handles=legend_elems, loc='best', frameon=True)

        plt.title(f"t-SNE highlight: {cls} (Q=green, G=yellow, others=blue)")
        plt.tight_layout()
        out_file = per_class_out_folder / f"{_slugify(cls)}.png"
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f">> Saved per-class t-SNE for '{cls}' to {out_file}")

    # consistent colors per class
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    colors = np.array([cls_to_idx[c] for c in all_labels])

    plt.figure(figsize=(9, 7))
    # plot queries as circles
    idx_q = np.array([s == 'Q' for s in all_split])
    plt.scatter(X2[idx_q, 0], X2[idx_q, 1], c=colors[idx_q], cmap="tab20",
                s=24, alpha=0.9, marker='o', linewidths=0.3)

    # plot galleries as crosses, same colors
    idx_g = ~idx_q
    plt.scatter(X2[idx_g, 0], X2[idx_g, 1], c=colors[idx_g], cmap="tab20",
                s=40, alpha=0.95, marker='x', linewidths=0.8)

    # legend (marker-only)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, label='Query (x per class)'),
        Line2D([0], [0], marker='x', linestyle='None', markersize=6, label='Gallery (x per class)'),
    ]
    plt.legend(handles=legend_elems, loc="best", frameon=True)
    plt.title(f"t-SNE of embeddings (2×x per class, {len(class_names)} classes)")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=200)
    plt.close()
    print(f">> Saved t-SNE to {tsne_path}")


def _minmax_along_axis(M: np.ndarray, axis: int) -> np.ndarray:
    """Min-max normalize along rows (axis=1) or columns (axis=0).
       Constant rows/cols become zeros."""
    mins = M.min(axis=axis, keepdims=True)
    maxs = M.max(axis=axis, keepdims=True)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    return (M - mins) / denom

def save_row_col_heatmaps(confusion: torch.Tensor, class_names, row_png, col_png, annotate=False):
    """
    Saves two images:
      - row_png: each row min-max normalized (lowest=yellow, highest=red)
      - col_png: each column min-max normalized (lowest=yellow, highest=red)
    """
    M = confusion.detach().cpu().numpy()

    # Row-normalized (axis=1)
    R = _minmax_along_axis(M, axis=1)  # values in [0,1] per row
    plt.figure(figsize=(max(6, 0.6*len(class_names)), max(5, 0.6*len(class_names))))
    im = plt.imshow(R, interpolation="nearest", vmin=0.0, vmax=1.0, cmap="autumn_r")  # yellow->red
    plt.title("Row-normalized similarity (per-row min–max)")
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    if annotate and len(class_names) <= 30:
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                plt.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(row_png, dpi=200)
    plt.close()
    print(f">> Saved row-normalized heatmap to {row_png}")

    # Column-normalized (axis=0)
    C = _minmax_along_axis(M, axis=0)  # values in [0,1] per column
    plt.figure(figsize=(max(6, 0.6*len(class_names)), max(5, 0.6*len(class_names))))
    im = plt.imshow(C, interpolation="nearest", vmin=0.0, vmax=1.0, cmap="autumn_r")  # yellow->red
    plt.title("Column-normalized similarity (per-column min–max)")
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    if annotate and len(class_names) <= 30:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                plt.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(col_png, dpi=200)
    plt.close()
    print(f">> Saved column-normalized heatmap to {col_png}")


    # Column-normalized (axis=0)
    C = _minmax_along_axis(M, axis=0)  # values in [0,1] per column
    plt.figure(figsize=(max(6, 0.6*len(class_names)), max(5, 0.6*len(class_names))))
    im = plt.imshow(C, interpolation="nearest", vmin=0.0, vmax=1.0, cmap="autumn_r")  # yellow->red
    plt.title("Column-normalized similarity (per-column min–max)")
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    if annotate and len(class_names) <= 30:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                plt.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(col_png, dpi=200)
    plt.close()
    print(f">> Saved column-normalized heatmap to {col_png}")

def save_confusion_png(confusion: torch.Tensor, class_names, out_file, annotate=False):
    M = confusion.cpu().numpy()
    plt.figure(figsize=(max(6, 0.6*len(class_names)), max(5, 0.6*len(class_names))))
    im = plt.imshow(M, interpolation="nearest", vmin=-1.0, vmax=1.0)  # cosine in [-1,1]
    plt.title("Mean cosine similarity (queries vs galleries)")
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.tight_layout()

    if annotate and len(class_names) <= 30:  # avoid clutter with many classes
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)

    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f">> Saved confusion heatmap to {out_file}")


def main():
    x = 10

    pretrained_weights = Path('./assets/weights/25-08-29T11-49-28_340k.pth')

    type_str, images_root = 'video_faces', Path(r'C:\Workspace\ChimpanzeesThesis\faces_images\individual_video_faces_dataset')
    # type_str, images_root = 'ccr_faces', Path(r"D:\faces_dataset_CCR_and_stils_zoo\train")

    # images_root = Path(r"C:\Workspace\ChimpanzeesThesis\faces_images\individual_faces_dataset")

    output_folder = Path('./logs') / f'{time.strftime("%Y%m%d_%H%M%S")}_{type_str}_x{x}'
    output_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_transforms = get_model(pretrained_weights, device)

    # add time for uniqueness
    out_json = f"results.json"
    confusion_png = f"confusion.png"
    rng = random.Random(42)

    # 1) discover class folders and images
    def is_img(p):
        return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    classes = OrderedDict()
    for clsdir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        imgs = [p for p in sorted(clsdir.iterdir()) if is_img(p)]
        if imgs:
            classes[clsdir.name] = imgs
    if not classes:
        raise RuntimeError(f"No class folders with images found under: {images_root}")

    class_names = list(classes.keys())

    # 2) sample 2*x per class, with replacement if needed, and split to queries/galleries
    def sample_with_replacement(paths, k):
        if len(paths) >= k:
            return rng.sample(paths, k)
        print('issue', paths[0])
        return [rng.choice(paths) for _ in range(k)]

    queries_by_class, galleries_by_class = {}, {}
    for cname, paths in classes.items():
        picks = sample_with_replacement(paths, 2 * x)
        queries_by_class[cname] = picks[:x]
        galleries_by_class[cname] = picks[x:]

    # 3) embed helper reusing your existing code; cache per-path
    @torch.no_grad()
    def embed_path(p: Path) -> torch.Tensor:
        if p not in _emb_cache:
            img = Image.open(p).convert("RGB")
            tens = img_transforms(img).unsqueeze(0)
            emb = get_embedding(model, tens, device)  # your function
            _emb_cache[p] = emb.squeeze(0).cpu()
        return _emb_cache[p]

    _emb_cache: dict[Path, torch.Tensor] = {}

    # also embed the single query image passed via --query_path if you want to include it later
    # (not used in the class-eval logic below)

    # 4) stack embeddings per class (queries and galleries)
    Q = {c: torch.stack([embed_path(p) for p in queries_by_class[c]], dim=0) for c in class_names}  # (x, D)
    G = {c: torch.stack([embed_path(p) for p in galleries_by_class[c]], dim=0) for c in class_names}  # (x, D)

    tsne_path = output_folder / "tsne_2x_per_class.png"
    tsne(tsne_path, Q, G, class_names)

    # 5) build confusion matrix: mean cosine similarity of queries(c_i) vs galleries(c_j)
    C = len(class_names)
    confusion = torch.zeros(C, C, dtype=torch.float32)
    for i, ci in enumerate(class_names):
        for j, cj in enumerate(class_names):
            sims = Q[ci] @ G[cj].T  # (x, x) because embeddings are L2-normalized
            confusion[i, j] = sims.mean().item()

    # 6) simple top-1 classification by averaging similarity to each class gallery set
    per_query_records = []
    correct = 0
    per_class_tot = defaultdict(int)
    per_class_cor = defaultdict(int)

    for i, ci in enumerate(class_names):
        q_embs = Q[ci]  # (x, D)
        # compute mean sim to each class's gallery set for all queries in this class
        class_means = []
        for cj in class_names:
            class_means.append((q_embs @ G[cj].T).mean(dim=1, keepdim=True))  # (x,1)
        class_means = torch.cat(class_means, dim=1)  # (x, C)

        preds = class_means.argmax(dim=1).tolist()
        for k in range(q_embs.size(0)):
            pred_idx = preds[k]
            pred_class = class_names[pred_idx]
            is_correct = int(pred_class == ci)
            correct += is_correct
            per_class_tot[ci] += 1
            per_class_cor[ci] += is_correct
            per_query_records.append({
                "true_class": ci,
                "pred_class": pred_class,
                "scores": {class_names[j]: float(class_means[k, j].item()) for j in range(C)},
                "query_path": str(queries_by_class[ci][k]),
            })

    overall_accuracy = correct / max(1, sum(per_class_tot.values()))
    per_class_accuracy = {c: per_class_cor[c] / max(1, per_class_tot[c]) for c in class_names}

    # 7) save everything to JSON
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(images_root),
        "x": x,
        "classes": class_names,
        "queries_by_class": {c: [str(p) for p in queries_by_class[c]] for c in class_names},
        "galleries_by_class": {c: [str(p) for p in galleries_by_class[c]] for c in class_names},
        "confusion_mean_cosine": [[float(v) for v in row] for row in confusion.tolist()],
        "per_query": per_query_records,
        "per_class_accuracy": {c: float(a) for c, a in per_class_accuracy.items()},
        "overall_accuracy": float(overall_accuracy),
    }
    (output_folder / out_json).write_text(json.dumps(payload, indent=4))

    # call it:
    save_confusion_png(confusion, class_names, out_file=output_folder / confusion_png, annotate=False)

# call after you compute `confusion` and have `class_names`
    save_row_col_heatmaps(confusion, class_names, row_png=output_folder / "confusion_rownorm.png", col_png=output_folder / "confusion_colnorm.png", annotate=False)
    print(f">> Saved JSON to {out_json}")
    print(f">> Overall top-1 accuracy: {overall_accuracy:.4f}")


if __name__ == '__main__':
    main()
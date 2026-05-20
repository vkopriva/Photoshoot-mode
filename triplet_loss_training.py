"""
triplet_loss_training.py
========================
Trains the DGCNN encoder using triplet loss with online batch-all mining.

Usage:
    # Train from scratch
    python triplet_loss_training.py --data_dir /path/to/ModelNet40_npy --output_dir ./checkpoints

    # Fine-tune from existing checkpoint
    python triplet_loss_training.py --data_dir /path/to/data --output_dir ./checkpoints \
        --checkpoint dgcnn_encoder_best.pth

    # Full custom configuration
    python triplet_loss_training.py \
        --data_dir    /path/to/data \
        --output_dir  ./checkpoints \
        --checkpoint  dgcnn_encoder_best.pth \
        --dgcnn_dir   /path/to/dgcnn_module \
        --epochs      100 \
        --batch_size  32 \
        --lr          1e-3 \
        --margin      0.2 \
        --n_points    128 256 512 1024

Outputs:
    checkpoints/dgcnn_triplet_best.pth   — best encoder by kNN accuracy
    checkpoints/dgcnn_triplet_last.pth   — final epoch weights
    checkpoints/training_log.json        — per-epoch metrics
    checkpoints/tsne_epoch_<N>.png       — t-SNE plots at save intervals
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Triplet Mining
# Vectorised implementation following Chierchia 2025 / Hermans et al. 2017.
# ─────────────────────────────────────────────────────────────────────────────

def get_pairs(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate all positive and negative pair masks for a batch."""
    positive = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive.fill_diagonal_(False)
    negative = labels.unsqueeze(0) != labels.unsqueeze(1)
    return positive, negative


def get_triplets(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate indices of all valid triplets (anchor, positive, negative)."""
    pos, neg = get_pairs(labels)
    pos = pos.unsqueeze(2)
    neg = neg.unsqueeze(1)
    triplets = pos & neg
    return torch.nonzero(triplets, as_tuple=True)


# ─────────────────────────────────────────────────────────────────────────────
# Triplet Loss — batch-all mining
# ─────────────────────────────────────────────────────────────────────────────

def triplet_loss_batch_all(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, dict]:
    """Batch-all triplet loss with easy triplet filtering."""
    anchor_idx, pos_idx, neg_idx = get_triplets(labels)

    n_total = len(anchor_idx)
    if n_total == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device), {
            "n_total": 0, "n_hard": 0, "n_semi_hard": 0, "fraction_active": 0.0
        }

    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    positive_dist = pairwise_dist[anchor_idx, pos_idx]
    negative_dist = pairwise_dist[anchor_idx, neg_idx]

    raw_loss = positive_dist - negative_dist + margin

    n_semi_hard = ((raw_loss > 0) & (raw_loss <= margin)).sum().item()
    n_hard = (raw_loss > margin).sum().item()

    active_loss = raw_loss[raw_loss > 0]

    stats = {
        "n_total": n_total,
        "n_hard": n_hard,
        "n_semi_hard": n_semi_hard,
        "fraction_active": len(active_loss) / n_total if n_total > 0 else 0.0,
    }

    if len(active_loss) == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device), stats

    return active_loss.mean(), stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """
    Loads .npy point clouds from per-class subdirectories:

        data_dir/
            chair/chair_001.npy   (N, 3)
            table/table_001.npy
            ...
    """

    def __init__(
        self,
        data_dir: Path,
        n_points: List[int],
        class_names: Optional[List[str]] = None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.augment = augment

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted([
                d.name for d in self.data_dir.iterdir() if d.is_dir()
            ])

        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.data_dir / cls
            for f in sorted(cls_dir.glob("*.npy")):
                self.samples.append((f, self.class_to_idx[cls]))

        print(f"[dataset] {len(self.class_names)} classes, "
              f"{len(self.samples)} total samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        pts = np.load(path).astype(np.float32)

        # Random density sampling — subsample first, then resample to fixed size
        n = random.choice(self.n_points)
        N = len(pts)
        pts = pts[np.random.choice(N, n, replace=(N < n))]

        # Normalise to unit sphere
        pts -= pts.mean(axis=0)
        d = np.linalg.norm(pts, axis=1).max()
        if d > 0:
            pts /= d

        if self.augment:
            # ── Online augmentations ──────────────────────────────────────
            # Based on established point cloud learning methods:
            #   - PointNet  (Qi et al., Stanford, 2017): jitter + dropout
            #   - DGCNN     (Wang et al., MIT, 2019):    rotation + scaling
            #   - PCT       (Guo et al., 2021):          translation + scaling
            #   - PointBERT (Yu et al., 2022):            rotation + scaling + jitter

            # 1. Random rotation around Z-axis
            #    Full rotation for learning rotation-invariant features
            #    (DGCNN / PointNet convention for object classification)
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T

            # 2. Anisotropic scaling (0.8–1.2 per axis)
            #    More effective than uniform scaling for learning shape features
            #    (PointContrast, Xie et al., Facebook AI, 2020)
            scale = np.random.uniform(0.8, 1.2, size=(1, 3)).astype(np.float32)
            pts *= scale

            # 3. Small random translation
            #    Simulates imperfect centering (PointNet, DGCNN)
            pts += np.random.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)

            # 4. Gaussian jitter — sensor noise
            #    sigma=0.01, clipped to [-0.05, 0.05] (PointNet convention)
            jitter = np.clip(
                np.random.normal(0, 0.01, pts.shape), -0.05, 0.05
            ).astype(np.float32)
            pts += jitter

            # 5. Random point dropout (10–30%)
            #    Simulates occlusion and partial views
            #    (PointDrop, Cheng et al., 2022)
            drop_rate = np.random.uniform(0.1, 0.3)
            keep_n = max(int(len(pts) * (1 - drop_rate)), 1)
            keep_idx = np.random.choice(len(pts), keep_n, replace=False)
            pts = pts[keep_idx]

        # Always resample to max point count so all tensors have equal size
        out_size = max(self.n_points)
        pts = pts[np.random.choice(len(pts), out_size, replace=(len(pts) < out_size))]

        # Return (N, 3) — DGCNN transposes internally
        return torch.from_numpy(pts).contiguous(), label


# ─────────────────────────────────────────────────────────────────────────────
# Model — L2 norm baked into forward()
# ─────────────────────────────────────────────────────────────────────────────

class NormalisedDGCNN(nn.Module):
    """DGCNN with L2 normalisation baked into forward pass."""
    def __init__(self, embed_dim: int, k: int):
        super().__init__()
        self.encoder = DGCNN(embed_dim=embed_dim, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return F.normalize(emb, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalised embeddings for an entire dataset split."""
    encoder.eval()
    all_emb, all_lbl = [], []

    for pts, labels in loader:
        pts = pts.to(device)
        emb = encoder(pts)
        all_emb.append(emb.cpu().numpy())
        all_lbl.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    return np.vstack(all_emb), np.concatenate(all_lbl)


def knn_accuracy(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    val_emb: np.ndarray,
    val_lbl: np.ndarray,
    k: int = 5,
) -> float:
    """kNN classification accuracy — mirrors FAISS retrieval at runtime."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_emb, train_lbl)
    preds = knn.predict(val_emb)
    return accuracy_score(val_lbl, preds)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
):
    """t-SNE visualisation of the embedding space."""
    print(f"  [t-SNE] Projecting {len(embeddings)} embeddings to 2D...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", len(class_names))

    for idx, cls in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(idx)], s=8, alpha=0.7, label=cls)

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=6, ncol=2, markerscale=3)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [t-SNE] Saved → {save_path}")


def plot_training_curves(training_log: list, best_knn_acc: float, save_path: Path):
    """Plot loss and kNN accuracy over epochs."""
    epochs_list = [e["epoch"] for e in training_log]
    losses = [e["train_loss"] for e in training_log]
    accs = [e["knn_accuracy"] for e in training_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    val_losses = [e.get("val_loss", None) for e in training_log]
    ax1.plot(epochs_list, losses, "b-", linewidth=1.5, label="Train")
    if val_losses[0] is not None:
        ax1.plot(epochs_list, val_losses, "r-", linewidth=1.5, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Triplet Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_list, [a * 100 for a in accs], "g-", linewidth=1.5)
    ax2.axhline(y=best_knn_acc * 100, color="r", linestyle="--", alpha=0.5,
                label=f"Best: {best_knn_acc*100:.1f}%")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("kNN Accuracy (%)")
    ax2.set_title("Validation kNN Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [curves] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    margin: float,
    device: torch.device,
) -> Tuple[float, dict]:
    """One full training epoch."""
    encoder.train()
    total_loss = 0.0
    total_hard = 0
    total_semi = 0
    n_batches = 0

    for pts, labels in loader:
        pts = pts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = encoder(pts)
        loss, stats = triplet_loss_batch_all(embeddings, labels, margin)

        if stats["n_total"] == 0:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_hard += stats["n_hard"]
        total_semi += stats["n_semi_hard"]
        n_batches += 1

    return total_loss / max(n_batches, 1), {
        "hard_triplets": total_hard,
        "semi_triplets": total_semi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DGCNN encoder with triplet loss (batch-all mining)."
    )

    # Paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with per-class subdirectories of .npy files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Where to save checkpoints, logs, t-SNE plots")
    parser.add_argument("--dgcnn_dir", type=str, default=None,
                        help="Directory containing dgcnn.py (default: same dir as this script)")
    parser.add_argument("--light", action="store_true",
                        help="Use DGCNN-Light (fewer layers, smaller k) for large batch training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Existing .pth to fine-tune from (omit to train from scratch)")

    # Data
    parser.add_argument("--n_points", type=int, nargs="+", default=[128, 256, 512, 1024],
                        help="Point counts sampled per object (default: 128 256 512 1024)")
    parser.add_argument("--val_split", type=float, default=0.2)

    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--k_neighbors", type=int, default=20)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32, increase if GPU memory allows)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Triplet margin (default: 0.2 for L2-normalised embeddings)")

    # Evaluation & Saving
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint + t-SNE every N epochs")
    parser.add_argument("--knn_k", type=int, default=5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ── Import DGCNN ──────────────────────────────────────────────────────────
    dgcnn_dir = args.dgcnn_dir or str(Path(__file__).parent)
    sys.path.insert(0, dgcnn_dir)
    global DGCNN
    if args.light:
        from dgcnn_light import DGCNN
    else:
        from dgcnn import DGCNN

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" TRIPLET LOSS TRAINING — DGCNN ENCODER")
    print("=" * 65)
    print(f"  Device         : {device}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  DGCNN module   : {dgcnn_dir}")
    print(f"  Point counts   : {args.n_points}  (variable density)")
    print(f"  Embed dim      : {args.embed_dim}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Learning rate  : {args.lr}")
    print(f"  Margin         : {args.margin}")
    print(f"  Mining         : batch-all (hard + semi-hard, easy filtered)")
    print(f"  L2 norm        : baked into encoder forward()")
    print(f"  Evaluation     : kNN accuracy + t-SNE")
    print(f"  Checkpoint     : {args.checkpoint or 'training from scratch'}")
    print("=" * 65 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = PointCloudDataset(
        data_dir=Path(args.data_dir),
        n_points=args.n_points,
        augment=True,
    )

    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[data] Train : {len(train_dataset)} samples")
    print(f"[data] Val   : {len(val_dataset)} samples\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = NormalisedDGCNN(args.embed_dim, args.k_neighbors).to(device)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        try:
            encoder.load_state_dict(state)
        except RuntimeError:
            encoder.encoder.load_state_dict(state)
        print(f"[model] Loaded weights from {args.checkpoint}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[model] Parameters: {n_params:,}\n")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_knn_acc = 0.0
    training_log = []

    for epoch in range(1, args.epochs + 1):

        train_loss, stats = train_one_epoch(
            encoder, train_loader, optimizer, args.margin, device
        )
        scheduler.step()

        train_emb, train_lbl = get_embeddings(encoder, train_loader, device)
        val_emb, val_lbl = get_embeddings(encoder, val_loader, device)

        acc = knn_accuracy(train_emb, train_lbl, val_emb, val_lbl, k=args.knn_k)

        # Validation loss (triplet loss on val embeddings, no gradient)
        val_emb_t = torch.from_numpy(val_emb).to(device)
        val_lbl_t = torch.from_numpy(val_lbl).to(device)
        val_loss, _ = triplet_loss_batch_all(val_emb_t, val_lbl_t, args.margin)
        val_loss = val_loss.item()

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "knn_accuracy": round(acc, 4),
            "hard_triplets": stats["hard_triplets"],
            "semi_triplets": stats["semi_triplets"],
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        training_log.append(log_entry)

        print(
            f"Epoch {epoch:>4}/{args.epochs}  |  "
            f"Loss: {train_loss:.4f}  |  "
            f"Val loss: {val_loss:.4f}  |  "
            f"kNN acc: {acc*100:.1f}%  |  "
            f"Hard: {stats['hard_triplets']:,}  "
            f"Semi: {stats['semi_triplets']:,}  |  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if acc > best_knn_acc:
            best_knn_acc = acc
            torch.save(encoder.state_dict(), output_dir / "dgcnn_triplet_best.pth")
            print(f"  → Best model saved (kNN acc={best_knn_acc*100:.1f}%)")

        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"dgcnn_triplet_epoch{epoch:04d}.pth"
            torch.save(encoder.state_dict(), ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")

            plot_tsne(
                val_emb, val_lbl,
                class_names=full_dataset.class_names,
                title=f"Embedding Space (t-SNE) — Epoch {epoch} | kNN {acc*100:.1f}%",
                save_path=output_dir / f"tsne_epoch_{epoch:04d}.png",
            )

    # ── Final outputs ─────────────────────────────────────────────────────────
    torch.save(encoder.state_dict(), output_dir / "dgcnn_triplet_last.pth")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    val_emb, val_lbl = get_embeddings(encoder, val_loader, device)
    plot_tsne(
        val_emb, val_lbl,
        class_names=full_dataset.class_names,
        title=f"Final Embedding Space (t-SNE) | Best kNN {best_knn_acc*100:.1f}%",
        save_path=output_dir / "tsne_final.png",
    )

    plot_training_curves(training_log, best_knn_acc, output_dir / "training_curves.png")

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best kNN accuracy : {best_knn_acc*100:.1f}%")
    print(f"  Best model        : {output_dir / 'dgcnn_triplet_best.pth'}")
    print(f"  Final model       : {output_dir / 'dgcnn_triplet_last.pth'}")
    print(f"  Training log      : {output_dir / 'training_log.json'}")
    print(f"  Training curves   : {output_dir / 'training_curves.png'}")
    print(f"  t-SNE plots       : {output_dir}/tsne_*.png")
    print("=" * 65)
    print("\nNext: run expand_faiss_FINAL.py with dgcnn_triplet_best.pth")


if __name__ == "__main__":
    main()
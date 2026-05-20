#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import faiss

# Adjust this if dgcnn.py is elsewhere
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dgcnn_light import DGCNN


SCRIPT_DIR = Path(__file__).resolve().parent
N_POINTS = 1024
EMBED_DIM = 256
K_NEIGHBORS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def process_cluster(og_points: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """Resample cluster to a fixed number of points."""
    pts = og_points.astype(np.float32)

    N = pts.shape[0]
    if N == 0:
        raise ValueError("Cannot process an empty cluster.")

    idx = np.random.choice(N, n_points, replace=(N < n_points))
    pts = pts[idx]

    return pts.astype(np.float32)


def cluster_points(
    points: np.ndarray,
    eps: float = 0.2,
    min_samples: int = 5,
    min_cluster_size: int = 40,
    max_cluster_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Run DBSCAN clustering and filter clusters by size.

    Returns:
        raw_clusters: valid cluster point clouds in original metric coordinates
        processed_clusters: resampled cluster point clouds for model input
        remapped_labels: per-point labels with valid clusters remapped to
                         0..N-1 and noise set to -1
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if len(points) == 0:
        return [], [], np.empty((0,), dtype=np.int32)

    if verbose:
        print(
            f"[clustering] Running DBSCAN on {len(points):,} points "
            f"(eps={eps}, min_samples={min_samples})"
        )

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)

    unique_labels = sorted(l for l in np.unique(labels) if l != -1)
    if verbose:
        print(f"[clustering] Raw clusters found: {len(unique_labels)}")

    raw_clusters: List[np.ndarray] = []
    processed_clusters: List[np.ndarray] = []
    valid_old_labels: List[int] = []

    for old_label in unique_labels:
        mask = labels == old_label
        cluster = points[mask]
        size = len(cluster)

        if size < min_cluster_size:
            if verbose:
                print(
                    f"[clustering] Rejecting cluster {old_label}: "
                    f"{size} points < min_cluster_size={min_cluster_size}"
                )
            continue

        if max_cluster_size is not None and size > max_cluster_size:
            if verbose:
                print(
                    f"[clustering] Rejecting cluster {old_label}: "
                    f"{size} points > max_cluster_size={max_cluster_size}"
                )
            continue

        raw_clusters.append(cluster.astype(np.float32))
        processed_cluster = process_cluster(cluster, n_points=N_POINTS)
        processed_clusters.append(processed_cluster)
        valid_old_labels.append(old_label)

    remap = {old_label: new_label for new_label, old_label in enumerate(valid_old_labels)}
    remapped_labels = np.full(labels.shape, -1, dtype=np.int32)

    for old_label, new_label in remap.items():
        remapped_labels[labels == old_label] = new_label

    if verbose:
        print(f"[clustering] Valid clusters after size filters: {len(raw_clusters)}")
    return raw_clusters, processed_clusters, remapped_labels


def estimate_bounding_box(points: np.ndarray) -> Dict[str, float]:
    """Compute an axis-aligned bounding box (AABB) for a raw cluster."""
    if len(points) == 0:
        raise ValueError("Cannot estimate a bounding box for an empty cluster.")

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    dimensions = max_coords - min_coords

    return {
        "min_x": float(min_coords[0]),
        "max_x": float(max_coords[0]),
        "min_y": float(min_coords[1]),
        "max_y": float(max_coords[1]),
        "min_z": float(min_coords[2]),
        "max_z": float(max_coords[2]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "width": float(dimensions[0]),
        "depth": float(dimensions[1]),
        "height": float(dimensions[2]),
        "num_points": int(len(points)),
    }


def estimate_bounding_boxes(clusters: List[np.ndarray]) -> List[Dict[str, float]]:
    """Compute AABBs for raw, unnormalized clusters."""
    return [estimate_bounding_box(cluster) for cluster in clusters]


def normalize_points(pts: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Normalize a cluster by centering it and scaling by the maximum radius.
    """
    if len(pts) == 0:
        return pts.astype(np.float32)

    original_center = pts.mean(axis=0)
    pts = pts - original_center

    d = np.max(np.linalg.norm(pts, axis=1))

    if verbose:
        print(f"[normalization] Normalizing cluster with {len(pts)} points")
        print(
            f"  Original centre: "
            f"({original_center[0]:+.3f}, {original_center[1]:+.3f}, {original_center[2]:+.3f})"
        )
        print(f"  Max distance from centre: {d:.3f} m")

    if d > 0:
        pts = pts / d

    return pts.astype(np.float32)


def load_encoder(encoder_path: Path) -> torch.nn.Module:
    """Load pretrained DGCNN encoder."""
    encoder = DGCNN(embed_dim=EMBED_DIM, k=K_NEIGHBORS).to(DEVICE)
    state_dict = torch.load(encoder_path, map_location=DEVICE)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def classify_clusters(
    clusters: List[np.ndarray],
    encoder: torch.nn.Module,
    index: faiss.Index,
    class_labels: List[str],
    threshold: float = 0.7,
    k: int = 5,
    verbose: bool = False,
) -> List[Tuple[str, float]]:
    """Encode each cluster → kNN query → majority vote class + confidence."""
    classifications: List[Tuple[str, float]] = []

    if len(class_labels) == 0:
        raise ValueError("class_labels is empty.")

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    k = min(k, index.ntotal)

    for i, cluster_pts in enumerate(clusters):
        if verbose:
            print(f"\n[classification] Cluster {i}: {len(cluster_pts)} points → encoding")

        pts = normalize_points(cluster_pts.copy(), verbose=verbose)

        # DGCNN expects shape (B, N, 3), e.g. (1, 1024, 3)
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = encoder(pts_t).detach().cpu().numpy().squeeze(0).astype(np.float32)

        # FAISS expects float32
        distances, indices = index.search(emb.reshape(1, -1), k)

        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(class_labels)]
        if not valid_indices:
            label = "Not a class"
            confidence = 0.0
            classifications.append((label, confidence))
            if verbose:
                print(f"  → {label} (conf: {confidence:.3f})")
            continue

        neighbor_labels = [class_labels[idx] for idx in valid_indices]
        vote_counts = Counter(neighbor_labels)
        top_class, top_votes = vote_counts.most_common(1)[0]
        confidence = top_votes / len(valid_indices)

        label = top_class if confidence >= threshold else "Not a class"
        classifications.append((label, confidence))

        if verbose:
            print(f"  Neighbor labels: {neighbor_labels}")
            print(f"  → {label} (conf: {confidence:.3f})")

    return classifications


def print_summary(points: np.ndarray, labels: np.ndarray, n_clusters: int) -> None:
    """
    Print a concise per-cluster summary to the terminal.
    """
    total = len(points)
    if total == 0:
        print("\n" + "=" * 55)
        print("CLUSTERING SUMMARY")
        print("=" * 55)
        print("  Total input points : 0")
        print("  Clustered points   : 0")
        print("  Noise points       : 0")
        print("  Valid clusters     : 0")
        print("=" * 55 + "\n")
        return

    noise = int((labels == -1).sum())

    print("\n" + "=" * 55)
    print("CLUSTERING SUMMARY")
    print("=" * 55)
    print(f"  Total input points : {total:,}")
    print(f"  Clustered points   : {total - noise:,}  ({100 * (total - noise) / total:.1f}%)")
    print(f"  Noise points       : {noise:,}  ({100 * noise / total:.1f}%)")
    print(f"  Valid clusters     : {n_clusters}")
    print("-" * 55)
    print(f"  {'Cluster':<12} {'Points':>8}  {'Centre (x, y, z)'}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 28}")

    for cid in range(n_clusters):
        mask = labels == cid
        center = points[mask].mean(axis=0)
        print(
            f"  Cluster {cid:<4}  {mask.sum():>8,}  "
            f"({center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f})"
        )

    print("=" * 55 + "\n")


def print_classification_summary(classifications: List[Tuple[str, float]]) -> None:
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    if not classifications:
        print("No clusters classified.")
    else:
        for i, (label, conf) in enumerate(classifications):
            print(f"Cluster {i:2d}: {label:20s} (conf: {conf:.3f})")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Processor class
# ─────────────────────────────────────────────────────────────────────────────

class ClusterProcessor:
    def __init__(
        self,
        eps: float = 0.2,
        min_samples: int = 5,
        min_cluster_size: int = 40,
        max_cluster_size: Optional[int] = None,
        visualize: bool = False,
        save_fig: Optional[str] = None,
        point_size: float = 1.0,
        max_normalized_plots: int = 6,
        show_plots: bool = True,
        encoder: Optional[torch.nn.Module] = None,
        index: Optional[faiss.Index] = None,
        class_labels: Optional[List[str]] = None,
        threshold: float = 0.7,
        k: int = 5,
        verbose: bool = False,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.visualize = visualize
        self.save_fig = save_fig
        self.point_size = point_size
        self.max_normalized_plots = max_normalized_plots
        self.show_plots = show_plots

        self.encoder = encoder
        self.index = index
        self.class_labels = class_labels
        self.threshold = threshold
        self.k = k
        self.verbose = verbose

    def _visualize_clusters(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        raw_clusters: List[np.ndarray],
        normalized_clusters: List[np.ndarray],
        bounding_boxes: List[Dict[str, float]],
        classifications: List[Tuple[str, float]],
        frame_idx: Optional[int] = None,
    ) -> None:
        """
        Render:
        1) Up to `max_normalized_plots` normalized clusters, each on its own 3D axis
        2) The original clustered point cloud in a separate figure
        """
        n_clusters = len(raw_clusters)
        cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
        title_suffix = f" (frame {frame_idx})" if frame_idx is not None else ""

        n_show = min(n_clusters, self.max_normalized_plots)

        if n_show == 0:
            fig1 = plt.figure(figsize=(6, 4))
            ax = fig1.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No valid clusters detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="grey",
            )
            ax.axis("off")
        else:
            ncols = min(3, n_show)
            nrows = int(np.ceil(n_show / ncols))
            fig1 = plt.figure(figsize=(6 * ncols, 5 * nrows))

            for cid in range(n_show):
                cluster = normalized_clusters[cid]
                ax = fig1.add_subplot(nrows, ncols, cid + 1, projection="3d")
                colour = cmap(cid)

                ax.scatter(
                    cluster[:, 0],
                    cluster[:, 1],
                    cluster[:, 2],
                    c=[colour],
                    s=self.point_size,
                    alpha=0.7,
                )

                ax.set_title(f"Normalized Cluster {cid}\n{len(cluster)} points")
                ax.set_xlabel("X (normalized)")
                ax.set_ylabel("Y (normalized)")
                ax.set_zlabel("Z (normalized)")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
                ax.set_zlim(-1.05, 1.05)

            if n_clusters > self.max_normalized_plots:
                fig1.suptitle(
                    f"Normalized Clusters{title_suffix} "
                    f"(showing first {self.max_normalized_plots} of {n_clusters})",
                    fontsize=14,
                )
            else:
                fig1.suptitle(f"Normalized Clusters{title_suffix}", fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig2 = plt.figure(figsize=(10, 8))
        ax3d = fig2.add_subplot(111, projection="3d")

        noise_mask = labels == -1
        if noise_mask.any():
            ax3d.scatter(
                points[noise_mask, 0],
                points[noise_mask, 1],
                points[noise_mask, 2],
                c="black",
                s=self.point_size * 0.5,
                alpha=0.15,
                label="Noise",
            )

        for cid in range(n_clusters):
            mask = labels == cid
            colour = cmap(cid)
            ax3d.scatter(
                points[mask, 0],
                points[mask, 1],
                points[mask, 2],
                c=[colour],
                s=self.point_size,
                alpha=0.7,
                label=f"Cluster {cid}",
            )

            bbox = bounding_boxes[cid]
            min_x, max_x = bbox["min_x"], bbox["max_x"]
            min_y, max_y = bbox["min_y"], bbox["max_y"]
            min_z, max_z = bbox["min_z"], bbox["max_z"]

            corners = np.array([
                [min_x, min_y, min_z], [max_x, min_y, min_z],
                [max_x, max_y, min_z], [min_x, max_y, min_z],
                [min_x, min_y, max_z], [max_x, min_y, max_z],
                [max_x, max_y, max_z], [min_x, max_y, max_z],
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            for start, end in edges:
                ax3d.plot(
                    [corners[start, 0], corners[end, 0]],
                    [corners[start, 1], corners[end, 1]],
                    [corners[start, 2], corners[end, 2]],
                    color=colour,
                    linewidth=1.5,
                    alpha=0.95,
                )

            if cid < len(classifications):
                cls_label, conf = classifications[cid]
                label_text = f"C{cid}: {cls_label} ({conf:.2f})"
            else:
                label_text = f"C{cid}"

            ax3d.text(
                bbox["center_x"],
                bbox["center_y"],
                bbox["max_z"] + max(0.05, 0.05 * max(bbox["height"], 1.0)),
                label_text,
                fontsize=8,
            )

        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        ax3d.set_title(
            f"Original Point Cloud{title_suffix}\n"
            f"{n_clusters} cluster(s)  |  "
            f"{noise_mask.sum()} noise pts  |  "
            f"{(~noise_mask).sum()} clustered pts"
        )

        if n_clusters <= 10:
            ax3d.legend(loc="upper left", fontsize=7, markerscale=3)

        plt.figure(fig2.number)
        plt.tight_layout()

        if self.save_fig:
            root, ext = os.path.splitext(self.save_fig)
            if not ext:
                ext = ".png"

            suffix = f"_frame_{frame_idx:06d}" if frame_idx is not None else ""
            norm_path = f"{root}{suffix}_normalized{ext}"
            orig_path = f"{root}{suffix}_original{ext}"

            fig1.savefig(norm_path, dpi=150, bbox_inches="tight")
            fig2.savefig(orig_path, dpi=150, bbox_inches="tight")
            print(f"[visualization] Normalized figure saved → {norm_path}")
            print(f"[visualization] Original figure saved   → {orig_path}")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

    def run(
        self,
        points: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run clustering + normalization + classification on an in-memory point cloud.
        """
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

        if self.verbose:
            print(f"\n[clustering] Input points: {len(points):,}")
            print(
                "[clustering] Parameters — "
                f"eps={self.eps}, "
                f"min_samples={self.min_samples}, "
                f"min_cluster_size={self.min_cluster_size}, "
                f"max_cluster_size={self.max_cluster_size}"
            )

        raw_clusters, processed_clusters, labels = cluster_points(
            points,
            eps=self.eps,
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            verbose=self.verbose,
        )

        bounding_boxes = estimate_bounding_boxes(raw_clusters)

        normalized_clusters: List[np.ndarray] = []
        for cluster in processed_clusters:
            normalized_clusters.append(normalize_points(cluster.copy(), verbose=self.verbose))

        if self.verbose:
            print_summary(points, labels, len(normalized_clusters))

        classifications: List[Tuple[str, float]] = []
        if self.encoder is not None and self.index is not None and self.class_labels is not None:
            classifications = classify_clusters(
                clusters=processed_clusters,
                encoder=self.encoder,
                index=self.index,
                class_labels=self.class_labels,
                threshold=self.threshold,
                k=self.k,
                verbose=self.verbose,
            )
            if self.verbose:
                print_classification_summary(classifications)
        else:
            if self.verbose:
                print("[classification] Skipped: encoder/index/class_labels not provided.")

        if self.visualize:
            self._visualize_clusters(
                points=points,
                labels=labels,
                raw_clusters=raw_clusters,
                normalized_clusters=normalized_clusters,
                bounding_boxes=bounding_boxes,
                classifications=classifications,
                frame_idx=frame_idx,
            )

        return {
            "frame_idx": frame_idx,
            "input_points": points,
            "labels": labels,
            "raw_clusters": raw_clusters,
            "processed_clusters": processed_clusters,
            "clusters": raw_clusters,
            "normalized_clusters": normalized_clusters,
            "bounding_boxes": bounding_boxes,
            "classifications": classifications,
            "num_clusters": len(normalized_clusters),
            "num_noise_points": int((labels == -1).sum()) if len(labels) else 0,
        }

    def run_from_file(
        self,
        input_path: str,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a .npy point cloud from disk, then run clustering.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        points = np.load(input_path)
        print(f"[clustering] Loaded {len(points):,} points from {input_path}")

        result = self.run(points=points, frame_idx=frame_idx)
        result["input_path"] = str(input_path)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DBSCAN clustering + per-cluster normalization + DGCNN/FAISS classification"
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to foreground .npy point cloud with shape (N, 3)")
    parser.add_argument("--visualize", action="store_true", help="Show clustering visualizations")
    parser.add_argument("--eps", type=float, default=0.2, help="DBSCAN neighbourhood radius in metres")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum points to form a cluster core")
    parser.add_argument("--min_cluster_size", type=int, default=40, help="Minimum points for a valid cluster")
    parser.add_argument("--max_cluster_size", type=int, default=None, help="Maximum points for a valid cluster")
    parser.add_argument("--save_fig", type=str, default=None, help="Optional output path prefix for saved figures")
    parser.add_argument("--point_size", type=float, default=1.0, help="Matplotlib scatter marker size")
    parser.add_argument("--max_normalized_plots", type=int, default=6, help="Maximum number of normalized cluster subplots to show")

    parser.add_argument("--index_dir", type=str, required=True, help="Directory containing library.index and labels.json")
    parser.add_argument("--encoder", type=str, required=True, help="Path to DGCNN encoder weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum confidence for predicted class")
    parser.add_argument("--k", type=int, default=5, help="Top-k FAISS neighbors for voting")

    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / "library.index"
    labels_path = index_dir / "labels.json"
    encoder_path = Path(args.encoder)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Class label file not found: {labels_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")

    print(f"[faiss] Loading index: {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"[faiss] Loaded {index.ntotal} vectors")

    with open(labels_path, "r") as f:
        class_labels = json.load(f)
    print(f"[faiss] Loaded {len(class_labels)} class labels")

    print(f"[encoder] Loading encoder weights: {encoder_path}")
    encoder = load_encoder(encoder_path)

    processor = ClusterProcessor(
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        visualize=args.visualize,
        save_fig=args.save_fig,
        point_size=args.point_size,
        max_normalized_plots=args.max_normalized_plots,
        show_plots=True,
        encoder=encoder,
        index=index,
        class_labels=class_labels,
        threshold=args.threshold,
        k=args.k,
    )

    result = processor.run_from_file(args.input_path)

    print("\nDone.")
    print(
        {
            "input_path": result.get("input_path"),
            "num_clusters": result["num_clusters"],
            "num_noise_points": result["num_noise_points"],
            "num_input_points": len(result["input_points"]),
            "classifications": result["classifications"],
        }
    )


if __name__ == "__main__":
    main()